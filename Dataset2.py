from torch.utils.data import Dataset, DataLoader
import natsort
import torch
import os
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.utils import save_image
import cv2
import numpy as np


class CustomDataSet(Dataset):
    def __init__(self, main_dir_vis, main_dir_ir, transform):
        self.main_dir_vis = main_dir_vis
        self.main_dir_ir = main_dir_ir
        self.transform = transform
        all_imgs_vis = os.listdir(main_dir_vis)
        all_imgs_ir = os.listdir(main_dir_ir)
        self.total_imgs_vis = natsort.natsorted(all_imgs_vis)
        self.total_imgs_ir = natsort.natsorted(all_imgs_ir)

    def __len__(self):
        return len(self.total_imgs_ir)

    def __getitem__(self, idx):
        vis_img_loc = os.path.join(self.main_dir_vis, self.total_imgs_vis[idx])
        # image_vis = Image.open(vis_img_loc).convert('L')                      # convert RGB to Grayscale
        image_vis = Image.open(vis_img_loc)                                     # loading RGB image
        width, height = image_vis.size
        left = (width - (45/64)* width) / 2                                     # Calculate the left coordinate for cropping
        right = (width + (45/64)* width) / 2                                    # Calculate the right coordinate for cropping
        top = 0                                                                 # Top coordinate remains unchanged
        bottom = height                                                         # Bottom coordinate remains unchanged
        cropped_image = image_vis.crop((left, top, right, bottom))
        resized_image = cropped_image.resize((640,512))
        tensor_image_vis = self.transform(resized_image)
        

        # vis_label_loc = os.path.join(self.label_dir_vis, self.total_imgs_vis[idx])
        ir_img_loc = os.path.join(self.main_dir_ir, self.total_imgs_ir[idx])
        # image_ir = Image.open(ir_img_loc).convert('L')
        image_ir = Image.open(ir_img_loc)

        ## converting to array for MfNET evaluation 
        # img2_array = np.array(image_ir)
        # print(f"img2_array shape {img2_array.shape}")
        # # Create a new array with three channels
        # three_channel_array = np.zeros((img2_array.shape[0], img2_array.shape[1], 3), dtype=np.uint8)

        # # Assign the single-channel values to the first channel
        # three_channel_array[:, :, 0] = img2_array
        # three_channel_array[:, :, 1] = img2_array
        # three_channel_array[:, :, 2] = img2_array

        # # Convert the array back to an image
        # result_image = Image.fromarray(three_channel_array)
        # image_ir = result_image
        # image_ir1 = ImageOps.pad(image_ir, (3840,2160), color="black")
        tensor_image_ir = self.transform(image_ir)
        return ({'image_vis' : tensor_image_vis, 'image_ir' : tensor_image_ir})