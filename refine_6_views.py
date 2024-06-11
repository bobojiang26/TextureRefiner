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

import argparse
from omegaconf import OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configs/base.yaml', help="path to the yaml config file")
args, extras = parser.parse_known_args()
opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

# print('Start rendering')

class RefTex:
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

    def refine_image(self, img, prompt):
        # stable diffusion refiner
        from diffusers import StableDiffusionXLImg2ImgPipeline, ControlNetModel
        seed = 12
        generator = torch.manual_seed(seed)
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe = pipe.to("cuda")

        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")

        pipe.controlnet = controlnet

        init_image = Image.fromarray(img.astype(np.uint8))
        # prompt = "a photo of a fox doll, real world, high quality, clear facial features, detailed"
        image = pipe(prompt=prompt + 'photo-realistic, high quality, detailed textures, 8k, ultrasharp', negative_prompt='art, painting, low quality, blurry, depth of field, out of focus', image=init_image,
                     strength = 0.5, num_inference_steps = 50, generator = generator
                     ).images[0]
        image.save('refined.png')

        # # stable diffusion super resolution 
        # from diffusers import StableDiffusionUpscalePipeline
        # model_id = "stabilityai/stable-diffusion-x4-upscaler"
        # pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
        # pipeline = pipeline.to("cuda")

        # image = np.array(image)
        # image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)


        # # model_id = "stabilityai/stable-diffusion-x4-upscaler"
        # # pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
        # # pipeline = pipeline.to("cuda")
        # # upscaled_image = pipeline(prompt="High quality, rich details, sharp, clear facial features, with more distinct boundaries in areas of color transitions, " + prompt, image=[image], num_inference_steps=20, guidance_scale=7.5).images[0]
        

        # from upscale_image import upscale_image
        # upscaled_image = upscale_image(img = image, rows = 1, cols = 1, seed = -1, prompt = 'A photorealistic portrait of ' + prompt + ', highly detailed facial features, sharp and clear edges, vibrant colors, and crisp outlines', negative_prompt='jpeg artifacts, lowres, bad quality',
        #                             xformers = True, cpu_offload = True, attention_slicing = True)
        
        # upscaled_image.save('upscaled_image.png')
        
        upscaled_image = np.array(image).astype(np.float32)

        return upscaled_image



    def refine(self):

        self.initialize(keep_ori_albedo=True)

        vers = [-15]
        hors = [180] 

        for ver, hor in tqdm.tqdm(zip(vers, hors), total=len(vers)):
            pose = orbit_camera(ver, hor, self.cam.radius)

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
            
            # cv2.imwrite('rendered_image.png', out['image'].cpu().numpy()*255)
            
            image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]
            # print(image.squeeze(0).permute(1,2,0).shape)

            # viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
            viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
            
            # add pose information to the prompt
            # if not opt.text_dir:
            #     text_embeds = self.guidance_embeds
            # else:
            #     # pose to view dir
            #     ver, hor, _ = undo_orbit_camera(pose)
            #     if ver <= -60: d = 'top'
            #     elif ver >= 60: d = 'bottom'
            #     else:
            #         if abs(hor) < 30: d = 'front'
            #         elif abs(hor) < 90: d = 'side'
            #         else: d = 'back'
            #     text_embeds = self.guidance_embeds[d]

            # refine rendered image
            rendered_image = np.array(image.squeeze(0).permute(1,2,0).cpu() * 255)
            cv2.imwrite('rendered_image.png', rendered_image[:,:,::-1])
            refined_image = self.refine_image(rendered_image, self.opt.prompt)
            
            refined_image = refined_image.astype(np.float32) / 255
            refined_image = torch.from_numpy(refined_image).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)

            # project-texture mask
            proj_mask = out['alpha'] > 0 & (out['viewcos'] > self.opt.cos_thresh) # [H, W, 1]

            # erose the project mask
            proj_mask_np = proj_mask.squeeze(2).cpu().numpy().astype(np.uint8)
            kernel_size = 10
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # cv2.imwrite('mask.png', proj_mask_np * 255)
            eroded_proj_mask_np = cv2.erode(proj_mask_np, kernel, iterations=1)
            # cv2.imwrite('eroded_mask.png', eroded_proj_mask_np * 255)
            proj_mask = torch.from_numpy(eroded_proj_mask_np).unsqueeze(2)

            proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
            uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')

            uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
            refined_image = refined_image.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]
            
            cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, refined_image, min_resolution=128, return_count=True)
            # cv2.imwrite('rendered_images/'+str(count)+'_rendered.png', output)

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
    def update_mesh_albedo(self):
        mask = self.cnt.squeeze(-1) > 0
        cur_albedo = self.albedo.clone()
        cur_albedo[mask] /= self.cnt[mask].repeat(1, 3)
        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def initialize(self, keep_ori_albedo=False):

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
    

    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.save_path)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
        return path
    

    def run(self):
        self.refine()
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

    reftex = RefTex(opt)

    reftex.run()
