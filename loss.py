import numpy as np
# from skimage.metrics import structural_similarity
from ssim import SSIM
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms


def _ssim(image1, image2, window_size=11, sigma=1.5):
    # compute mean values
    mu1 = F.avg_pool2d(image1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(image2, window_size, stride=1, padding=window_size//2)

    # compute variances and covariances
    sigma1_2 = F.avg_pool2d(image1**2, window_size, stride=1, padding=window_size//2) - mu1**2
    sigma2_2 = F.avg_pool2d(image2**2, window_size, stride=1, padding=window_size//2) - mu2**2
    sigma12 = F.avg_pool2d(image1 * image2, window_size, stride=1, padding=window_size//2) - mu1 * mu2

    # compute SSIM
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    ssim_map = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2) / ((mu1**2 + mu2**2 + C1) * (sigma1_2 + sigma2_2 + C2))

    # return mean SSIM value
    return torch.mean(ssim_map)


def calculate_ssim(image1, image2, patch_size=64, metric = 'ssim'):
    # make sure images are the same size
    min_height = min(image1.shape[2], image2.shape[2])
    min_width = min(image1.shape[3], image2.shape[3])
    image1 = image1[:min_height, :min_width, :]
    image2 = image2[:min_height, :min_width, :]

    # image1 = resize(image1, (512, 512))
    # image2 = resize(image2, (512, 512))

    # calculate the number of patches
    num_patches_height = image1.shape[2] // patch_size
    num_patches_width = image1.shape[3] // patch_size
    num_patches = num_patches_height * num_patches_width

    # initialize array to store SSIM values for each patch
    metric_values = np.empty((num_patches,))

    # loop over patches and calculate SSIM for each patch
    patch_index = 0
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            # calculate patch boundaries
            row_start = i * patch_size
            row_end = row_start + patch_size
            col_start = j * patch_size
            col_end = col_start + patch_size

            # extract patches from images
            patch1 = image1[:, :, row_start:row_end, col_start:col_end]
            patch2 = image2[:, :, row_start:row_end, col_start:col_end]

            if metric == 'ssim':
                # calculate SSIM for patch and store in array
                metric_values[patch_index] = _ssim(patch1, patch2)
            elif metric == 'psnr':
                metric_values[patch_index] = PSNR(patch1, patch2)
                
        

            patch_index += 1

    # calculate mean SSIM value for all patches
    metric_value = np.mean(metric_values)

    return metric_value


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Normalize the embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate and normalize the embeddings
        embeddings = torch.cat([z1, z2], dim=0)
        embeddings = F.normalize(embeddings, dim=1)
        batch_size = embeddings.shape[0]

        # Reshape the tensors for cosine similarity calculation
        embeddings = embeddings.view(batch_size, -1)  # (batch_size, channels * height * width)

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t())

        # Exclude the main diagonal from similarity calculations
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool)
        similarity_matrix = similarity_matrix[~mask].view(embeddings.size(0), -1)

        # Compute contrastive loss
        contrastive_loss = -torch.log_softmax(similarity_matrix / self.temperature, dim=1)
        contrastive_loss = contrastive_loss.diag().mean()

        return contrastive_loss



def test():
    x = torch.randn((8, 3, 512, 640))
    y = torch.randn((8, 3, 512, 640))
    # model = CrossAttention(in_channels=32)
    p , q = calculate_ssim(x,y,patch_size= 32, metric= 'ssim')
    print(f"ssim : {p}")


if __name__ == "__main__":
    test()