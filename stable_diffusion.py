import os
from transformers import logging
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class StableDiffusion(nn.Module):
    def __init__(
            self,
            device,
            fp16 = True,
            model_key = "runwayml/stable-diffusion-v1-5",
            control_mode = "depth_inpaint",
            lora_keys = [],
    ):
        super().__init__()


        self.device = device
        self.control_mode = control_mode
        self.dtype = torch.float16 if fp16 else torch.float32
        
        # create stablediffusionpipeline
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype)
        pipe.to(device)

        self.vae = pipe.vae 
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder 
        self.unet = pipe.unet

        # controlnet
        self.controlnet = {}
        self.controlnet_conditioning_scale = {}
        self.controlnet["depth_inpaint"] = ControlNetModel.from_pretrained("ashawkey/control_v11e_sd15_depth_aware_inpaint",torch_dtype=self.dtype).to(self.device)
        self.controlnet_conditioning_scale['depth_inpaint'] = 1.0

        self.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        del pipe

    
    def encode_imgs(self, imgs):
        # imgs * 2 before being encoded, and after decoding latent to the image, the image should be /2
        imgs = 2 * imgs - 1 
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents


    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    

    def __call__(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        guidance_rescale=0,
        control_images=None,
        latents=None,
        strength=0,
        refine_strength=0.8,
    ):

        # text_embedding and control_images to self.dtype
        text_embeddings = text_embeddings.to(self.dtype)
        for i in control_images:
            control_images[i] = control_images[i].to(self.dtype)

        # initialize the latents
        if latents == None:
            latents = torch.randn((text_embeddings.shape[0]//2, 4, height//8, width//8), dtype = self.dtype, device = self.device)


        # initialize self.scheduler and init_step
        self.scheduler.set_timesteps(num_inference_steps)
        init_step = 0

        # diffusion process loop    
        for i, t in enumerate(self.scheduler.timesteps[init_step:]):

            t = torch.tensor([t], dtype=self.dtype, device=self.device)

            # inpaint mask blend
            if "latents_mask" in control_images:
                if i < num_inference_steps * refine_strength:
                    mask_keep = 1 - control_images["latents_mask"]
                else:
                    mask_keep = control_images["latents_mask_keep"]

                latents_original = control_images["latents_original"]
                latents_original_noise = self.scheduler.add_noise(latents_original, torch.randn_like(latents_original), t)
                latents = latents * (1 - mask_keep) + latents_original_noise * mask_keep       

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # controlnet
            control_image = control_images["depth_inpaint"]
            control_image_input = torch.cat([control_image] * 2)

            noise_pred = 0
            down_samples, mid_sample = self.controlnet["depth_inpaint"](
                latent_model_input, t, encoder_hidden_states=text_embeddings, 
                        controlnet_cond=control_image_input, 
                        conditioning_scale=self.controlnet_conditioning_scale["depth_inpaint"],
                        return_dict=False
            )

            noise_pred_cur = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings, 
                        down_block_additional_residuals=down_samples, 
                        mid_block_additional_residual=mid_sample
            ).sample

            noise_pred = noise_pred + noise_pred_cur

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            if guidance_rescale > 0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        return imgs


        


    
