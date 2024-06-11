import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import cv2
import numpy as np

# 加载超分辨率模型
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
pipeline = pipeline.to("cuda")  # 如果你有CUDA支持，可以使用GPU进行加速

# 加载你的图像
input_image_path = "/home/zcb/self_code_training/InTeX_self/data/texture_9.png"  # 替换为你的图像路径
image = Image.open(input_image_path).convert("RGB")
image = np.array(image)
image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)

# add gaussian noise
# mean = 0
# sigma = 25
# gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
# noisy_image = cv2.add(image, gaussian_noise)
# cv2.imwrite("noisy_image.png", noisy_image)


# 处理图像
from upscale_image import upscale_image
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
# pipeline = pipeline.to("cuda")
# upscaled_image = pipeline(prompt="", image=[cur_albedo], num_inference_steps=50, guidance_scale=7.5).images[0]

upscaled_image = upscale_image(img = image, rows = 1, cols = 1, seed = -1, prompt = 'A fox doll, 8K, best quality, Highly detailed, physically-based rendering, Professional', negative_prompt='jpeg artifacts, lowres, bad quality, deblurred, noisy',
                                xformers = True, cpu_offload = True, attention_slicing = True)
upscaled_image = np.array(upscaled_image).astype(np.float32)

# Use realsergan to upscale the texture
# cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio)
# cur_albedo = cur_albedo.astype(np.float32) / 255


# 保存放大的图像
output_image_path = "/home/zcb/self_code_training/InTeX_self/data/texture_9_new.png"  # 替换为你想保存的路径
upscaled_image = Image.fromarray((upscaled_image).astype('uint8'))
upscaled_image.save(output_image_path)

print(f"图像已保存到 {output_image_path}")
