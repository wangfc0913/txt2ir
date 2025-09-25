import os
import sys
import pickle
import random
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from einops import rearrange
import torchvision.transforms as transforms
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "../../data"))


def get_image(img_path, img_type='vis', fine_size=512):
    assert img_path, "ERROR [get_image func]: img_path is None"

    img = None
    if img_type == 'vis':
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img.copy())
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    elif img_type == 'ir':
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)
        img = transforms.ToTensor()(img.copy()).float()
        img = transforms.Normalize([0.5], [0.5])(img)
    else:
        print(f'Warning: [get_image func] Unrecognized image type')
        exit(-1)

    if fine_size is not None:
        w_total = img.size(2)
        w = int(w_total)
        h = img.size(1)
        w_offset = max(0, (w - fine_size - 1) // 2)
        h_offset = max(0, (h - fine_size - 1) // 2)

        img = img[:, h_offset:h_offset + fine_size, w_offset:w_offset + fine_size]

    return np.array(img.permute(1, 2, 0).numpy()).astype(np.float32)


def get_mask(img_path, fine_size=512):
    assert img_path, "ERROR [get_mask func]: img_path is None"

    img = Image.open(img_path)
    img = transforms.ToTensor()(img.copy()).float()

    if fine_size is not None:
        w_total = img.size(2)
        w = int(w_total)
        h = img.size(1)
        w_offset = max(0, (w - fine_size - 1) // 2)
        h_offset = max(0, (h - fine_size - 1) // 2)

        img = img[:, h_offset:h_offset + fine_size, w_offset:w_offset + fine_size]

    return np.array(img.permute(1, 2, 0).numpy()).astype(np.float32)


class T2IRDataset(Dataset):
    def __init__(self, data_name=None, mode='train', fine_size=512):
        assert data_name is not None, "must specify the dataset name."
        assert mode in ['train', 'val', 'test'], "mode must be one of ['train', 'val', 'test']."
        self.data_path = os.path.join(data_dir, data_name, mode)
        self.fine_size = fine_size

        self.filenames = self.load_filenames()

    def load_filenames(self):
        filepath = os.path.join(self.data_path, 'filenames.pkl')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def load_caption(self, file_ame):
        filepath = os.path.join(self.data_path, 'captions')
        cap_path = '%s/%s_ir.txt' % (filepath, file_ame)
        with open(cap_path, 'rb') as f:
            np_lines = np.stack([[line.strip()] for line in f.readlines()])
        return np_lines

    def __getitem__(self, index):
        file_name = self.filenames[index]

        captions = self.load_caption(file_name)
        cap_ix = random.randint(0, captions.shape[0] - 1)
        caption = captions[cap_ix, :]
        caption = str(caption)

        ir_path = '%s/images/%s_ir.jpg' % (self.data_path, file_name)
        mask_path = '%s/images/%s_mask.jpg' % (self.data_path, file_name)
        ir_img = get_image(img_path=ir_path, img_type='ir', fine_size=self.fine_size)
        mask_img = get_mask(img_path=mask_path, fine_size=self.fine_size)

        return {'image': ir_img, 'mask': mask_img, 'caption': caption, 'file_name': file_name}

    def __len__(self):
        return len(self.filenames)


class T2IRTrain(T2IRDataset):
    def __init__(self, data_name='txt2irDataset', mode='train', fine_size=512):
        super().__init__(data_name=data_name, mode=mode, fine_size=fine_size)


class T2IRValidation(T2IRDataset):
    def __init__(self, data_name='txt2irDataset', mode='test', fine_size=512):
        super().__init__(data_name=data_name, mode=mode, fine_size=fine_size)
