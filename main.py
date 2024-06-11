import os
import cv2
import time
import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from cam_utils import orbit_camera, undo_orbit_camera, OrbitCamera
from mesh_renderer import Renderer

from scipy import ndimage
from kornia.morphology import dilation
from grid_put import mipmap_linear_grid_put_2d, linear_grid_put_2d, nearest_grid_put_2d

import kiui
# from diffusers import StableDiffusionUpscalePipeline

from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion


def dilate_image(image, mask, iterations):
    # image: [H, W, C], current image
    # mask: [H, W], region with content (~mask is the region to inpaint)
    # iterations: int

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    if mask.dtype != bool:
        mask = mask > 0.5

    inpaint_region = binary_dilation(mask, iterations=iterations)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    image[tuple(inpaint_coords.T)] = image[tuple(search_coords[indices[:, 0]].T)]
    return image

class InTex:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = opt.seed
        self.save_path = opt.save_path

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.buffer_overlay = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.buffer_out = None  # for 2D to 3D projection

        self.need_update = True  # update buffer_image
        self.need_update_overlay = True  # update buffer_overlay

        self.mouse_loc = np.array([0, 0])
        self.draw_mask = False
        self.draw_radius = 20
        self.mask_2d = np.zeros((self.W, self.H, 1), dtype=np.float32)

        # models
        self.device = torch.device("cuda")

        self.guidance = None
        self.guidance_embeds = None

        # renderer
        self.renderer = Renderer(self.device, opt)

        # input mesh
        if self.opt.mesh is not None:
            self.renderer.load_mesh(self.opt.mesh)

        # input text
        self.prompt = self.opt.posi_prompt + ', ' + self.opt.prompt
        self.negative_prompt = self.opt.nega_prompt

    def prepare_guidance(self):
        if self.guidance is None:
            from stable_diffusion import StableDiffusion
            self.guidance = StableDiffusion(device=self.device, model_key=self.opt.model_key, lora_keys=self.opt.lora_keys)
        print(f'[INFO] loaded guidance model!')

        print(f'[INFO] encoding prompt...')
        nega = self.guidance.get_text_embeds([self.negative_prompt])

        # self.guidance_embeds initialize
        if not self.opt.text_dir:
            posi = self.guidance.get_text_embeds([self.prompt])
            self.guidance_embeds = torch.cat([nega, posi], dim=0)
        else:
            self.guidance_embeds = {}
            posi = self.guidance.get_text_embeds([self.prompt])
            self.guidance_embeds['default'] = torch.cat([nega, posi], dim=0)
            for d in ['front', 'side', 'back', 'top', 'bottom']:
                posi = self.guidance.get_text_embeds([self.prompt + f', {d} view'])
                self.guidance_embeds[d] = torch.cat([nega, posi], dim=0)
        
    @torch.no_grad()
    def inpaint_view(self, pose, count):

        h = w = int(self.opt.texture_size)
        H = W = int(self.opt.render_resolution)
        # first render
        out = self.renderer.render(pose, self.cam.perspective, H, W)

        # valid crop region with fixed aspect ratio
        valid_pixels = out['alpha'].squeeze(-1).nonzero() # [N, 2]
        min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
        min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()
        
        size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
        h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
        w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

        min_h = int(h_start)
        min_w = int(w_start)
        max_h = int(min_h + size)
        max_w = int(min_w + size)

        # crop region is outside rendered image: do not crop at all.
        if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
            min_h = 0
            min_w = 0
            max_h = H
            max_w = W

        def _zoom(x, mode='bilinear', size=(H, W)):
            return F.interpolate(x[..., min_h:max_h+1, min_w:max_w+1], size, mode=mode)
        # print(out['image'].shape) #[1024, 1024, 3]
        image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]

        # output the image_array
        image_pil = Image.fromarray((image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
        # image_pil = Image.fromarray((out['alpha'].squeeze(2).cpu().numpy() * 255).astype('uint8'), 'L')
        image_pil.save(str(count)+'.png')

        # trimap: generate, refine, keep
        mask_generate = _zoom(out['cnt'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='nearest') < 0.1 # [1, 1, H, W]

        viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
        viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
        mask_refine = ((viewcos_old < viewcos) & ~mask_generate)

        mask_keep = (~mask_generate & ~mask_refine)

        mask_generate = mask_generate.float()
        mask_refine = mask_refine.float()
        mask_keep = mask_keep.float()


        mask_generate_blur = mask_generate
        
        # if no generate region, return 
        if not (mask_generate > 0.5).any():
            return
        
        control_images = {}

        image_generate = image.clone()

        # mask_keep先在channel上复制3次以来和rgb3通道匹配，接着mask_keep上小于0。5的也就是不是1的区域就都是要inpaint的区域（1是generate的区域），让要inpaint的区域全部赋值-1
        image_generate[mask_keep.repeat(1, 3, 1, 1) < 0.5] = -1 # -1 is inpaint region

        image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
        depth = _zoom(out['depth'].view(1, 1, H, W), size=(512, 512)).clamp(0, 1).repeat(1, 3, 1, 1) # [1, 3, H, W]
        
        # write the depth map
        # image_pil = Image.fromarray((depth.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
        # # image_pil = Image.fromarray((out['alpha'].squeeze(2).cpu().numpy() * 255).astype('uint8'), 'L')
        # image_pil.save(str(count)+'_depth.png')

        control_images['depth_inpaint'] = torch.cat([image_generate, depth], dim=1) # [1, 6, H, W]

        # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
        latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
        latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
        latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
        # latents_mask is latents_mask_generate
        control_images['latents_mask'] = latents_mask
        control_images['latents_mask_refine'] = latents_mask_refine
        control_images['latents_mask_keep'] = latents_mask_keep

        control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]
        

        if not self.opt.text_dir:
            text_embeds = self.guidance_embeds
        else:
            # pose to view dir
            ver, hor, _ = undo_orbit_camera(pose)
            if ver <= -60: d = 'top'
            elif ver >= 60: d = 'bottom'
            else:
                if abs(hor) < 30: d = 'front'
                elif abs(hor) < 90: d = 'side'
                else: d = 'back'
            text_embeds = self.guidance_embeds[d]
        
        # 对应到StableDiffusion的__call__()函数：
        # rgbs is the depth-aware-inpainting result
        rgbs = self.guidance(text_embeds, height=512, width=512, control_images=control_images, refine_strength=self.opt.refine_strength).float()
        # rgbs.shape = [1, 3, 512, 512]
        
        # performing upscaling (assume 2/4/8x)
        if rgbs.shape[-1] != W or rgbs.shape[-2] != H:
            scale = W // rgbs.shape[-1]
            rgbs = rgbs.detach().cpu().squeeze(0).permute(1, 2, 0).contiguous().numpy()
            rgbs = (rgbs * 255).astype(np.uint8)
            # perform realesrgan to upscale the rgb
            rgbs = kiui.sr.sr(rgbs, scale=scale)
            rgbs = rgbs.astype(np.float32) / 255
            rgbs = torch.from_numpy(rgbs).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        
        # apply mask to make sure non-inpaint region is not changed
        # rgbs = rgbs * (1 - mask_keep) + image * mask_keep

        # project-texture mask
        proj_mask = (out['alpha'] > 0) & (out['viewcos'] > self.opt.cos_thresh)  # [H, W, 1]
        proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
        uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')

        uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
        rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]
        # uvs.shape: N * 2


        # mipmap_linear_grid_put_2d is the projection function
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True)
        # cur_albedo.shape = [1024, 1024, 3]


        self.backup()

        mask = cur_cnt.squeeze(-1) > 0
        self.albedo[mask] += cur_albedo[mask]
        self.cnt[mask] += cur_cnt[mask]

        # update mesh texture for rendering
        self.update_mesh_albedo()
        
        # update viewcos cache
        viewcos = viewcos.view(-1, 1)[proj_mask]
        cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)

    @torch.no_grad()
    def backup(self):
        self.backup_albedo = self.albedo.clone()
        self.backup_cnt = self.cnt.clone()
        self.backup_viewcos_cache = self.renderer.mesh.viewcos_cache.clone()

    @torch.no_grad()
    def restore(self):
        self.albedo = self.backup_albedo.clone()
        self.cnt = self.backup_cnt.clone()
        self.renderer.mesh.cnt = self.cnt
        self.renderer.mesh.viewcos_cache = self.backup_viewcos_cache.clone()
        self.update_mesh_albedo()
    
    @torch.no_grad()
    def update_mesh_albedo(self):
        mask = self.cnt.squeeze(-1) > 0
        cur_albedo = self.albedo.clone()
        cur_albedo[mask] /= self.cnt[mask].repeat(1, 3)
        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def dilate_texture(self):
        h = w = int(self.opt.texture_size)

        self.backup()

        mask = self.cnt.squeeze(-1) > 0
        
        ## dilate texture
        mask = mask.view(h, w)
        mask = mask.detach().cpu().numpy()

        self.albedo = dilate_image(self.albedo, mask, iterations=int(h*0.2))

        # write the texture map
        # print('albedo.shape: ', self.albedo.shape)
        # image_pil = Image.fromarray((self.albedo.cpu().numpy()).astype('uint8'))
        # # image_pil = Image.fromarray((out['alpha'].squeeze(2).cpu().numpy() * 255).astype('uint8'), 'L')
        # image_pil.save('texture.png')

        self.cnt = dilate_image(self.cnt, mask, iterations=int(h*0.2))
        
        self.update_mesh_albedo()
    
    @torch.no_grad()
    def deblur(self, ratio=2):
        h = w = int(self.opt.texture_size)

        self.backup()

        # overall deblur by LR then SR
        kiui.vis.plot_image(self.albedo)
        cur_albedo = self.renderer.mesh.albedo.detach().cpu().numpy()
        cur_albedo = (cur_albedo * 255).astype(np.uint8)
        cur_albedo = cv2.resize(cur_albedo, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)

        # Use the super resolution stable diffusion model to upscale the image
        from upscale_image import upscale_image
        # model_id = "stabilityai/stable-diffusion-x4-upscaler"
        # pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
        # pipeline = pipeline.to("cuda")
        # upscaled_image = pipeline(prompt="", image=[cur_albedo], num_inference_steps=50, guidance_scale=7.5).images[0]

        # upscaled_image = upscale_image(img = cur_albedo, rows = 2, cols = 2, seed = -1, prompt = '8k, photography, cgi, unreal engine, octane render, best quality', negative_prompt='jpeg artifacts, lowres, bad quality',
        #                                xformers = True, cpu_offload = True, attention_slicing = True)
        # cur_albedo = np.array(upscaled_image).astype(np.float32) / 255

        # Use realsergan to upscale the texture
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio)
        cur_albedo = cur_albedo.astype(np.float32) / 255


        kiui.vis.plot_image(cur_albedo)
        cur_albedo = torch.from_numpy(cur_albedo).to(self.device)

        # enhance quality by SD refine...
        # kiui.vis.plot_image(albedo.detach().cpu().numpy())
        # text_embeds = self.guidance_embeds if not self.opt.text_dir else self.guidance_embeds['default']
        # albedo = self.guidance.refine(text_embeds, albedo.permute(2,0,1).unsqueeze(0).contiguous(), strength=0.8).squeeze(0).permute(1,2,0).contiguous()
        # kiui.vis.plot_image(albedo.detach().cpu().numpy())

        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def generate(self):

        self.initialize(keep_ori_albedo=False)

        if self.opt.camera_path == 'default':
            vers = [-15] * 8 + [-89.9, 89.9] + [45]
            hors = [180, 45, -45, 90, -90, 135, -135, 0] + [0, 0] + [0]
        elif self.opt.camera_path == 'front':
            vers = [0] * 8 + [-89.9, 89.9] + [45]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif self.opt.camera_path == 'top':
            vers = [0, -45, 45, -89.9, 89.9] + [0] + [0] * 6
            hors = [0] * 5 + [180] + [45, -45, 90, -90, 135, -135]
        elif self.opt.camera_path == 'side':
            vers = [0, 0, 0, 0, 0] + [-45, 45, -89.9, 89.9] + [-45, 0]
            hors = [0, 45, -45, 90, -90] + [0, 0, 0, 0] + [180, 180]
        else:
            raise NotImplementedError(f'camera path {self.opt.camera_path} not implemented!')
        
        start_t = time.time()
        print(f'[INFO] start generation...')
        count = 0
        for ver, hor in tqdm.tqdm(zip(vers, hors), total=len(vers)):
            # render image
            pose = orbit_camera(ver, hor, self.cam.radius)
            self.inpaint_view(pose, count)
            count += 1

            # preview
            self.need_update = True
            self.test_step()
        
        torch.cuda.empty_cache()
        self.dilate_texture()
        self.deblur()

        torch.cuda.synchronize()
        end_t = time.time()

        print(f'[INFO] finished generation in {end_t - start_t:.3f}s!')

        self.need_update = True

    
    @torch.no_grad()
    def initialize(self, keep_ori_albedo=False):
        self.prepare_guidance()

        h = w = int(self.opt.texture_size)
        self.albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
        self.cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)
        self.viewcos_cache = - torch.ones((h, w, 1), device=self.device, dtype=torch.float32)

        if keep_ori_albedo:
            # albedo map (diffuse map)
            self.albedo = self.renderer.mesh.albedo.clone()
            # count
            self.cnt += 1 # set to 1
            # view cosine cache
            self.viewcos_cache *= -1 # set to 1


        # self.render initialize
        self.renderer.mesh.albedo = self.albedo
        self.renderer.mesh.cnt = self.cnt 
        self.renderer.mesh.viewcos_cache = self.viewcos_cache

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update and not self.need_update_overlay:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha', 'viewcos', 'viewcos_cache', 'cnt']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
            
            if self.mode in ['normal', 'rot_normal']:
                buffer_image = (buffer_image + 1) / 2

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()

            self.buffer_out = out

            self.need_update = False
        
        # should update overlay
        if self.need_update_overlay:
            buffer_overlay = np.zeros_like(self.buffer_overlay)

            # draw mask 2d
            buffer_overlay += self.mask_2d * 0.5
            
            self.buffer_overlay = buffer_overlay
            self.need_update_overlay = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)


    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.save_path)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
        return path
        
    def run(self):
        self.generate()
        self.save_model()


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/base.yaml', help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    
    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    print(opt)

    intex = InTex(opt)

    intex.run()
