import os
import torch
from torch.utils import data
from torchvision import transforms, utils

import numpy as np
from PIL import Image

class MF_dataset_test(data.Dataset):
    
    def __init__(self, data_dir):

        txt = 'test.txt'
        
        with open(os.path.join(data_dir , txt), 'r') as file:
            self.names = [name.strip() for idx, name in enumerate(file)]

        self.data_dir = data_dir
        self.image_folder = 'images'
        self.label_folder = 'labels'

    def __getitem__(self, index):

        img_name = self.names[index]

        image_path = os.path.join(self.data_dir, self.image_folder, img_name + '.png')
        image = Image.open(image_path)
        image = np.asarray(image, dtype=np.float32)  


        image = image.transpose((2, 0, 1)) / 255  
        image = torch.tensor(image)

        th_image = image[3]
        th_image = th_image.unsqueeze(0)

        label_path = os.path.join(self.data_dir, self.label_folder, img_name + '.png')
        label = Image.open(label_path)
        label = np.asarray(label, dtype=np.int64)

        label = torch.tensor(label)
        
        return th_image, label

    def __len__(self):
        return len(self.names)