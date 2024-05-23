import torch
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import nibabel as nib
import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torchio as tio
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from custom_transforms import *
import cv2
from monai.transforms import AsDiscrete

class tumor_dataset(Dataset):
    def __init__(self, csv_path, transform=None):

        self.csv_path = csv_path
        self.transforms = transform
        self.threshold = AsDiscrete(threshold=0.5)
        
        df = pd.read_csv(self.csv_path)
        
        self.input_list = []
        self.mask_list = []
        self.label_list = []

        for index, row in df.iterrows():
            self.input_list.append(row['INPUT_PATH'])
            self.mask_list.append(row['TARGET_PATH'])
            self.label_list.append(row['LABELS'])

    def __len__(self):
        if len(self.input_list) == len(self.mask_list):
            return len(self.mask_list)
        else:
            return "error"
        
    def resample_dicom_to_numpy(self,input_dicom_path, new_spacing):

        # Read the DICOM file
        ds = pydicom.dcmread(input_dicom_path)

        # Extract original pixel spacing
        original_spacing = ds.PixelSpacing

        # Extract the image data
        image = ds.pixel_array
        image[image < 0] = 0

        # Calculate resize factor
        resize_factor = np.array(original_spacing) / np.array(new_spacing)

        # Calculate the new shape of the image
        new_shape = np.round(image.shape * resize_factor).astype(int)

        # Resample the image
        resampled_image = cv2.resize(image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)

        return resampled_image
            
    def preprocess(self,train_path,mask_path):
        

        input_img = self.resample_dicom_to_numpy(train_path,[0.625,0.625])
        epsilon = 1e-10
        min_val = np.min(input_img)
        max_val = np.max(input_img)
        input_img = (input_img - min_val) / (max_val - min_val+epsilon)
        
        target_img = self.resample_dicom_to_numpy(mask_path,[0.625,0.625])
        epsilon = 1e-10
        min_val = np.min(target_img)
        max_val = np.max(target_img)
        target_img = (target_img - min_val) / (max_val - min_val+epsilon)

        return input_img, target_img



    def __getitem__(self, idx):

        input_path = self.input_list[idx]
        target_path = self.mask_list[idx]
        label = self.label_list[idx]

        input_img,mask_img = self.preprocess(input_path,target_path)

        mask_img = np.where(mask_img > 0.5, 1, 0)

        if self.transforms:
            transformed = self.transforms(image=input_img, mask = mask_img)
            input_img = transformed['image']
            mask_img = transformed['mask']

        return input_img,mask_img.unsqueeze(0),label


if __name__ == '__main__':
    import albumentations as A
    train_transform = A.Compose([
        A.Resize(512,512),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5), 
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])
    dataset = tumor_dataset(csv_path="***",transform=train_transform)
    train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=0)
    sample = next(iter(train_loader))
    print(len(dataset))
    print(torch.min(sample[0]))
