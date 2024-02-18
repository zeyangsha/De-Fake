import json
import torchvision.transforms as transforms
import torchvision
import PIL.Image as Image
import os
import torch
import torch.nn as nn
from pathlib import Path
import PIL.Image as Image
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
import pandas as pd

class real(torch.utils.data.Dataset):
    def __init__(self,realroot,size,transform=None):
        self.transform = transforms.Compose([
            transforms.Resize((size,size)),
            #RandAugment(2, 14),
            #transforms.CenterCrop((size,size)),
            transforms.ToTensor()
        ])
        dataDir='your dir'
        dataType='val2014'
        self.annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
        self.coco=COCO(self.annFile)
        self.imgIds_list=sorted(self.coco.getImgIds())
    
    def __getitem__(self,item):
        imgIds = self.coco.getImgIds(imgIds = [self.imgIds_list[item]])
        img = self.coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
        I = io.imread(img['coco_url'])
        real_image = Image.fromarray(I).convert('RGB')
        real_image = self.transform(real_image)
        annIds = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(annIds)
        
        label = 0
        return real_image,label,anns[0]['caption']
        
    def __len__(self):
        return len(self.imgIds_list)
        
class realflickr(torch.utils.data.Dataset):
    def __init__(self,realroot,size,transform=None):
        self.transform = transforms.Compose([
            transforms.Resize((size,size)),
            #RandAugment(2, 14),
            #transforms.CenterCrop((size,size)),
            transforms.ToTensor()
        ])
        annotations = pd.read_table('your dir', sep='\t', header=None,
                            names=['image', 'caption'])
        self.prompt_list = np.array(annotations['caption'][::5])
        self.image_list = np.array(annotations['image'][::5])
        
    def __getitem__(self,item):
        real_image = Image.open('your dir')
        prompts = self.prompt_list[item]
        label = 0
        real_image = self.transform(real_image)
        return real_image,label,prompts
        
    def __len__(self):
        return len(self.image_list)


class fakereal(torch.utils.data.Dataset):
    def __init__(self,fakeroot,size,transform=None):
        self.transform = transforms.Compose([
            transforms.Resize((size,size)),
            #RandAugment(2, 14),
            #transforms.CenterCrop((size,size)),
            transforms.ToTensor()
        ])

        fake_images_path = Path(fakeroot)
        fake_images_list = list(fake_images_path.glob('*.png'))
        fake_images_list_str = [ str(x) for x in fake_images_list ]
        self.fake_images = fake_images_list_str
    
    def __getitem__(self,item):
        fake_image_path = self.fake_images[item]
        fake_image = Image.open(fake_image_path).convert('RGB')
        fake_image = self.transform(fake_image)
        label = 1
        prompts = fake_image_path.split('/')[-1].replace('-',' ').split('.png')[0]

        return fake_image,label,prompts
    
    def __len__(self):
        return len(self.fake_images)
        
        
        
        
class fakeclip(torch.utils.data.Dataset):
    def __init__(self,fakeroot,size,transforms=None):
        self.transform = transforms.Compose([
            transforms.Resize((size,size)),
            #RandAugment(2, 14),
            #transforms.CenterCrop((size,size)),
            transforms.ToTensor()
        ])

        fake_images_path = Path(fakeroot)
        fake_images_list = list(fake_images_path.glob('*.png'))
        fake_images_list_str = [ str(x) for x in fake_images_list ]
        self.fake_images = fake_images_list_str
    