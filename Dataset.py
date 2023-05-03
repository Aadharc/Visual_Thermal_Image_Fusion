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
    '''
    Custom Dataset class to load the dataset containing misaligned visual and thermal images with their corresponding
    truth label bounding boxes for human detection.

    main_dir_vis : directory for visual images
    main_dir_ir : directory for thermal images
    label_dir_vis : directory for visual images bboxes
    label_dir_ir : directory for thermal images bboxes
    transform : torchvision transformer for preprocessing

    '''
    def __init__(self, main_dir_vis, main_dir_ir, label_dir_vis, label_dir_ir, transform):
        self.main_dir_vis = main_dir_vis
        self.main_dir_ir = main_dir_ir
        self.label_dir_vis = label_dir_vis
        self.label_dir_ir = label_dir_ir
        self.transform = transform
        # list all the images
        all_imgs_vis = os.listdir(main_dir_vis)
        all_imgs_ir = os.listdir(main_dir_ir)
        all_label_vis = os.listdir(label_dir_vis)
        all_label_ir = os.listdir(label_dir_ir)

        # sorting the images to ensure training with right pair of therma and visual images
        self.total_imgs_vis = natsort.natsorted(all_imgs_vis)
        self.total_imgs_ir = natsort.natsorted(all_imgs_ir)
        self.total_label_vis = natsort.natsorted(all_label_vis)
        self.total_label_ir = natsort.natsorted(all_label_ir)
        self.vis_ground_truth = {}

        # Load the bounding box ground truth data for visual images
        for f in self.total_label_vis:
            if f.endswith('.txt'):
                img_name = f.split('.')[0]
                with open(os.path.join(self.label_dir_vis, f), 'r') as gt_file:
                    bboxes = []
                    for line in gt_file:
                        # bbox = list(map(int, line.strip().split(',')))
                        bbox = list(map(float, line.strip().split()))
                        bboxes.append(bbox)
                    self.vis_ground_truth[img_name] = bboxes
        # Load the bounding box ground truth data for thermal images
        self.thermal_ground_truth = {}
        # for f in os.listdir(thermal_gt_dir):
        for f in self.total_label_ir:
            if f.endswith('.txt'):
                img_name = f.split('.')[0]
                with open(os.path.join(self.label_dir_ir, f), 'r') as gt_file:
                    bboxes = []
                    for line in gt_file:
                        # bbox = list(map(int, line.strip().split(',')))
                        bbox = list(map(float, line.strip().split()))
                        bboxes.append(bbox)
                    self.thermal_ground_truth[img_name] = bboxes

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
        # image_ir = image_ir.resize((2700,2160))
        # image_ir1 = ImageOps.pad(image_ir, (3840,2160), color="black")
        tensor_image_ir = self.transform(image_ir)

        # getting the corresponding bboxes of visual and therma images
        ir_img_name = self.total_imgs_ir[idx].split('.')[0]
        vis_img_name = self.total_imgs_vis[idx].split('.')[0]
        thermal_bboxes = self.thermal_ground_truth[ir_img_name]
        visual_bboxes = self.vis_ground_truth[vis_img_name]
        return ({'image_vis' : tensor_image_vis, 'image_ir' : tensor_image_ir, 'target_vis' :(torch.tensor(visual_bboxes, dtype = torch.float32)).shape[0], 'target_ir' : (torch.tensor(thermal_bboxes, dtype=torch.float32)).shape[0]})


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((256,512),transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    # transform = transforms.Compose([transforms.Resize((512,512),transforms.InterpolationMode.BILINEAR)])
    dataset = CustomDataSet("/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/Data for GAN/Data/data_2/vis/val","/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/Data for GAN/Data/data_2/ir/val", transform)
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(y.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()

