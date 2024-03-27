import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import pydicom

class tumor_dataset(Dataset):
    def __init__(self, csv_path, transform=None):

        self.csv_path = csv_path
        self.transforms = transform
        
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
        
    def preprocess(self,train_path,mask_path):
        
        input_slice = pydicom.read_file(train_path)

        input_img = input_slice.pixel_array
        input_img[input_img < 0] = 0

        epsilon = 1e-10
        min_val = np.min(input_img)
        max_val = np.max(input_img)
        input_img = (input_img - min_val) / (max_val - min_val+epsilon)
        

        target_slice = pydicom.read_file(mask_path)
        target_img = target_slice.pixel_array
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

