from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

image = load_image(
    "/home/zcb/self_code_training/InTeX_self/2drefined_results/backpack/rendered_image.png"
)

image = np.array(image)
image = custom_blur_demo(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

canny_image.save('/home/zcb/self_code_training/InTeX_self/2drefined_results/backpack/canny_image.png')