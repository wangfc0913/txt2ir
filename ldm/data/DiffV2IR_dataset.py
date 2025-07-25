from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import pickle
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip_models.blip import blip_decoder
import json
import os

def load_demo_image(image_size,device,img_url):
    
    raw_image = Image.open(img_url).convert('RGB')   

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0)   
    return image


class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
    ):
        assert split in ("train", "val", "test")
        self.path = os.path.join(path, split)

        self.filenames = self.load_filenames()

        self.blip_model = blip_decoder(pretrained="./blip_models/model__base_caption.pth", image_size=512, vit='base')
        self.blip_model.eval()

    def load_filenames(self):
        filepath = os.path.join(self.path, 'filenames.pkl')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        name = self.filenames[i]

        path = '%s/images/%s_ir.jpg' % (self.path, name)

        ir_dir = '%s/ir/%s_ir.png' % (self.path, name)
        rgb_dir = '%s/rgb/%s_co.png' % (self.path, name)
        seg_dir = '%s/seg/%s.png' % (self.path, name)

        image = load_demo_image(image_size=512, device='cuda', img_url=rgb_dir)

        with torch.no_grad():
            caption = self.blip_model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        prompt = "turn the visible image of "+caption[0]+" into infrared"

        image_0 = Image.open(ir_dir)
        image_1 = Image.open(rgb_dir)
        image_2 = Image.open(seg_dir)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")
        image_2 = rearrange(2 * torch.tensor(np.array(image_2)).float() / 255 - 1, "h w c -> c h w")

        image_0, image_1, image_2 = torch.cat((image_0, image_1,image_2)).chunk(3)

        return dict(edited=image_0, edit=dict(c_concat1=image_1, c_concat2=image_2, c_crossattn=prompt))
