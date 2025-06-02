import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import os
from torchvision import transforms as tsfm
from PIL import Image
from torchvision.io import read_image 
from data.imgaug import GetTransforms
from data.utils import transform
np.random.seed(0)
import sys 
sys.path.append('/media/Datacenter_storage/')


class ImageDataset_Mayo(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._bias_header = None
        self._image_paths = []
        self._bias = []
        self._causal = []
        self._labels = []
        self._mode = mode
        # Define the indices of the columns you want to extract
        label_column_index = -6  # 

        # ##CKD and Chf
        # causal_columns_indices = [4,5]
      
        # bias_columns_indices = [2,3]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [header[label_column_index]]
            # self._bias_header = [header[i] for i in bias_columns_indices]
            # self._causal_header = [header[i] for i in causal_columns_indices]
            for line in f:
            #for i, line in enumerate(f):
                # Check if we've reached the 100th line
                #if i >= 20:
                  #  break  # Exit the loop
                fields = line.strip('\n').split(',')
                image_path = fields[-1]
                print(f"image_path: {image_path}")
                #print(f"ckd:{fields[5]}")
                label = float(fields[label_column_index])  # Single float value now
                # causal = [fields[i] for i in causal_columns_indices]
                # bias = [fields[i] for i in bias_columns_indices]
                self._image_paths.append(image_path)
                self._labels.append(label)
                # self._bias.append(bias)
                # self._causal.append(causal)
                
            
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image
    
    current_directory = os.getcwd()

    print("The script is running from:", current_directory)

    # Padding function to first make the image (1280, 1280)

    
    def __getitem__(self, idx):
        #print(f"image_path: {self._image_paths[idx]}")
        image_path = self._image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
        h,w,c = image.shape
        intermediate_size = (w, w)  # First padding target
        final_size = (512, 512)  # Final training size
        padded_image = pad_to_square(image, intermediate_size)

        final_image = cv2.resize(padded_image, final_size, interpolation=cv2.INTER_AREA)
        # save_path = os.path.join("/media/Datacenter_storage/jialu/003/ECGchange_VITCOAT/ECG_single/image_test", f"processed_image_{idx}.png")
        # cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))  # Convert back to BGR before saving


        # Normalize to [0, 1]
        max_v = final_image.max()
        min_v = final_image.min()
        image = (final_image - min_v) * (1 / (max_v - min_v))
        image = image.transpose(2, 0, 1) 
        # print(f"image: max={image.max()}, min={image.min()}")
        image = torch.tensor(image, dtype=torch.float32) 


        label = float(self._labels[idx])  # No longer an array
        # bias = np.array(self._bias[idx]).astype(np.float32)
        # causal = np.array(self._causal[idx]).astype(np.float32)
        path = self._image_paths[idx]
        
        # Structure the return based on the operation mode
        if self._mode in ['train', 'dev']:
            return image, label
        elif self._mode in ['test', 'heatmap']:
            return image, path, label
        else:
            raise Exception(f'Unknown mode: {self._mode}')
    

def pad_to_square(image, target_size):
    h, w, c = image.shape
    target_h, target_w = target_size

    # Compute required padding for height (Top & Bottom)
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top

    # Compute required padding for width (Left & Right) (should be 0 in your case)
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    # Apply padding
    image_padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                          mode='constant', constant_values=255)

    return image_padded
        



