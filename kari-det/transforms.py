import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

class ImageAug:
    def __init__(self, train):
        if train:
            self.aug = A.Compose([A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.5),
                                  #A.ShiftScaleRotate(p=0.5),
                                  A.RandomCrop(512, 512),
                                  ToTensorV2()])
        else:
            self.aug = A.Compose([#A.HorizontalFlip(p=0.5),
                                  #A.VerticalFlip(p=0.5),
                                  #A.ShiftScaleRotate(p=0.5),
                                  A.RandomCrop(512, 512),
                                  ToTensorV2()])
        
        
    def __call__(self, img, mask_img):
        transformed = self.aug(image=img, mask=mask_img)      
        return transformed['image'], transformed['mask']

    def remap_idxes(self, mask):
        mask = torch.where(mask >= 1000, mask.div(1000, rounding_mode='floor'), mask)
        for void_idx in self.void_idxes:
            mask[mask == void_idx] = self.ignore_idx
        for valid_idx in self.valid_idxes:
            mask[mask == valid_idx] = self.class_map[valid_idx]
        return mask

   
def get_transforms(train):
    transforms = ImageAug(train)
    return transforms