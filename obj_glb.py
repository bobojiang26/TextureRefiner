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

class ObjToGlb:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda")
        self.save_path = opt.save_path
        self.renderer = Renderer(self.device, opt)
        if self.opt.mesh is not None:
            self.renderer.load_mesh(self.opt.mesh)


    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.save_path)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
        return path
    

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/base.yaml', help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    
    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    print(opt)

    objtoglb = ObjToGlb(opt)

    objtoglb.save_model()
