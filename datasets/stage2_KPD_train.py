import os
import os.path as path
import torch
from torch.utils import data
from torchvision import transforms, utils

import numpy as np
from PIL import Image
from utils.augmentation import RandomNoise

class stage2_KP_dataset(data.Dataset):
    """
    rgb_img : BxCxHxW
    th_img : Bx1xHxW
    label : BxHxW
    """

    def __init__(self, data_dir, input_folder='pseudo_KP', transform=None, fake_folder='day2night'):

        with open(os.path.join(data_dir, 'filenames_KP', 'train_all_rgb.txt'), 'r') as file:
            self.rgb_names = [name.strip() for idx, name in enumerate(file)]
        
        with open(os.path.join(data_dir, 'filenames_KP', 'train_all_th.txt'), 'r') as file:
            self.th_names = [name.strip() for idx, name in enumerate(file)]

        assert len(self.rgb_names) == len(self.th_names)

        self.transform = transform 

        self.data_dir = data_dir
        self.inputs = input_folder
        self.fake_folder = fake_folder
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        name = (self.rgb_names[index]).replace('_visible', '', 1)
        name = name.split('.png')[0] 
        
        rgb_path = os.path.join(self.data_dir, self.inputs, 'all', name + '_rgb.png')
        rgb_image = Image.open(rgb_path)
        rgb_image = np.asarray(rgb_image, dtype=np.float32)  # HxWxC

        th_path = os.path.join(self.data_dir, self.inputs, 'all', name + '_th.png')
        th_image = Image.open(th_path).convert('L')
        th_image = np.asarray(th_image, dtype=np.float32)  # HxWxC

        pseudo_path = os.path.join(self.data_dir, self.inputs, 'all', name + '_pseudo.png')
        pseudo = Image.open(pseudo_path)
        pseudo = np.asarray(pseudo, dtype=np.int64)

        fake=False
        if self.transform is not None:
            for func in self.transform:
                rgb_image, th_image, pseudo, fake = func(rgb_image, th_image, pseudo, fake)
   
        rgb_image = rgb_image.transpose((2, 0, 1)) / 255 
        rgb_image = torch.tensor(rgb_image).float()
        rgb_image = self.normalize(rgb_image)
        th_image = th_image / 255
        th_image = torch.tensor(th_image).float()
        th_image = th_image.unsqueeze(0)
        
        pseudo = torch.tensor(pseudo)

        return rgb_image, th_image, pseudo, name

    def __len__(self):
        return len(self.rgb_names)
