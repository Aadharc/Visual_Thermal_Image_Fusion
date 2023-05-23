import cv2
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_mutual_information

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def SSIM(img1, img2):
    img1 = img1.permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().numpy()
    ssim = np.mean([compare_ssim(im1, im2, multichannel=True) for im1, im2 in zip(img1, img2)])
    return ssim

# def SSIM(img1, img2):
#     img1 = img1.permute(1, 2, 0).cpu().numpy()
#     img2 = img2.permute(1, 2, 0).cpu().numpy()
#     ssim = np.mean([compare_ssim(im1, im2, multichannel=True) for im1, im2 in zip(img1, img2)])
#     return ssim


def FF(img1, img2, fused):
    energy1 = torch.sum(img1 ** 2)
    energy2 = torch.sum(img2 ** 2)
    energyF = torch.sum(fused ** 2)
    ff = (energyF - energy1 - energy2) / (2 * torch.sqrt(energy1 * energy2))
    return ff.item()

# def NMI(img1, img2):
#     img1 = img1.view(-1).cpu().numpy()
#     img2 = img2.view(-1).cpu().numpy()
#     nmi = normalized_mutual_information(img1, img2)
#     return nmi

def NMI(img1, img2):
    img1 = img1.reshape(-1).cpu().numpy()
    img2 = img2.reshape(-1).cpu().numpy()
    nmi = normalized_mutual_information(img1, img2)
    return nmi

### Working for images not batches

# def calculate_patch_metrics(image1, image2, patch_size=32):
#     # Calculate the number of patches in the images
#     num_patches = (image1.shape[0] // patch_size) * (image1.shape[1] // patch_size)

#     # print(f"image1 shape, {image1.shape}")
    
#     # Initialize arrays to store the SSIM values for each patch
#     ssim_values = np.zeros(num_patches)
#     psnr_values = np.zeros(num_patches)
#     nmi_values = np.zeros(num_patches)
    
#     # Loop through each patch in the images
#     patch_index = 0
#     for i in range(0, image1.shape[0], patch_size):
#         for j in range(0, image1.shape[1], patch_size):
#             # Extract the patch from each image
#             patch1 = image1[i:i+patch_size, j:j+patch_size]
#             print(f"patch size, {patch1.shape}")
#             patch2 = image2[i:i+patch_size, j:j+patch_size]
            
#             # Calculate the SSIM for the patch and store the result
#             ssim_values[patch_index] = SSIM(patch1, patch2)
#             psnr_values[patch_index] = PSNR(patch1, patch2)
#             nmi_values[patch_index] = NMI(patch1, patch2)
            
#             # Increment the patch index
#             patch_index += 1
    
#     # Calculate the average SSIM for all patches
#     # avg_ssim = np.mean(ssim_values)
    
#     return psnr_values, ssim_values, nmi_values


## Working for batches

def calculate_patch_metrics(image1, image2, patch_size=128):
    # Calculate the number of patches in the images
    num_patches = (image1.shape[2] // patch_size) * (image1.shape[3] // patch_size)
    
    # Initialize arrays to store the SSIM values for each patch
    ssim_values = np.zeros(num_patches)
    psnr_values = np.zeros(num_patches)
    nmi_values = np.zeros(num_patches)
    
    # Loop through each patch in the images
    patch_index = 0
    for i in range(0, image1.shape[0], patch_size):
        for j in range(0, image1.shape[1], patch_size):
            # Extract the patch from each image
            patch1 = image1[:, :, i:i+patch_size, j:j+patch_size]
            patch2 = image2[:, :, i:i+patch_size, j:j+patch_size]
            
            # Calculate the SSIM for the patch and store the result
            ssim_values[patch_index] = SSIM(patch1, patch2)
            psnr_values[patch_index] = PSNR(patch1, patch2)
            nmi_values[patch_index] = NMI(patch1, patch2)
            
            # Increment the patch index
            patch_index += 1
    
    # Calculate the average SSIM for all patches
    # avg_ssim = np.mean(ssim_values)
    
    return psnr_values, ssim_values, nmi_values


# def calculate_metrics_with_patch(gen, val_loader, epoch, folder, device):
#     psnr_values_vis = []
#     ssim_values_vis = []
#     ff_values = []
#     nmi_values_vis = []

#     psnr_values_ir = []
#     ssim_values_ir = []
#     ff_values = []
#     nmi_values_ir = []

#     for batch in val_loader:
#         img1 = batch['image_vis']
#         img2 = batch['image_ir']
#         a = batch['target_vis']
#         b = batch['target_ir']
#         # masked_feat = mask_feat(x, y)
#         # mask_x, mask_y = masked_feat[2].to(config.DEVICE), masked_feat[3].to(config.DEVICE)
#         img1, img2 = img1.to(device), img2.to(device)
#         gen.eval()
#         # print(f"image size in main metric fun", {img1.shape})
#         with torch.no_grad():
#             # Perform image fusion
#             y_fake, x_a, y_a = gen(img1, img2)
#             # Calculate the metrics for Vis
#             psnr_vis, ssim_vis, nmi_vis = calculate_patch_metrics(img1, y_fake, patch_size = 128)
#             # print(f"psnr outpot shape from the patch metrics fun, {psnr_vis.shape}")

#             # Append the metric values to the lists
#             psnr_values_vis.append(psnr_vis)
#             ssim_values_vis.append(ssim_vis)
#             # ff_values.append(ff)
#             nmi_values_vis.append(nmi_vis)

#             # Calculate the metrics for ir
#             psnr_ir, ssim_ir, nmi_ir = calculate_patch_metrics(img2, y_fake, patch_size = 128)

#             # Append the metric values to the lists
#             psnr_values_ir.append(psnr_ir)
#             ssim_values_ir.append(ssim_ir)
#             nmi_values_ir.append(nmi_ir)

#     # Create a dictionary to store the metric values       
#     metrics = {'PSNR_VIS': psnr_values_vis,
#                'SSIM_VIS': ssim_values_vis,
#                'NMI_VIS': nmi_values_vis,
#             #    'FF': ff_values,
#                'PSNR_IR' : psnr_values_ir,
#                'SSIM_IR' : ssim_values_ir,
#                'NMI_IR' : nmi_values_ir}

#     # Create the folder if it doesn't exist
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     # Get the number of images and patches
#     num_images = len(metrics['PSNR_VIS'])
#     num_patches = len(metrics['PSNR_VIS'][0])


#     # Create a 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Create arrays to store the X, Y, and Z values
#     X = np.arange(num_images)
#     Y = np.arange(num_patches)
#     X, Y = np.meshgrid(X, Y)

#     # Plot each metric as a surface
#     for metric, values in metrics.items():
#         Z = np.array(values)
#         Z = np.reshape(Z, X.shape)
#         ax.plot_surface(X, Y, Z, cmap='viridis', label=metric)

#         # Set the axis labels and legend
#         ax.set_xlabel('Image Number')
#         ax.set_ylabel('Patch Number')
#         ax.set_zlabel('Metric Value')
#         plt.savefig(os.path.join(folder, f'metric_{metric}_{epoch}.png'))
#     # ax.legend()

#     # # Show the plot
#     # plt.show()

#     # Get the maximum length of the metric value lists
#     # max_len = max(len(x) for x in metrics.values())

#     # # Pad the shorter lists with NaN values
#     # for key, val in metrics.items():
#     #     if len(val) < max_len:
#     #         metrics[key] = val + [float('nan')] * (max_len - len(val))

#     # # Create a 3D plot
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')

#     # # Create arrays to store the X, Y, and Z values
#     # num_images = len(metrics['PSNR_VIS'])
#     # num_patches = max_len
#     # X = np.arange(num_images)
#     # Y = np.arange(num_patches)
#     # X, Y = np.meshgrid(X, Y)

#     # # Plot each metric as a surface
#     # for metric, values in metrics.items():
#     #     Z = np.array(values).reshape(num_images, num_patches)
#     #     ax.plot_surface(X, Y, Z, cmap='viridis', label=metric)

#     # # Set the axis labels and legend
#     # ax.set_xlabel('Image Number')
#     # ax.set_ylabel('Patch Number')
#     # ax.set_zlabel('Metric Value')
#     # ax.legend()

#     return metrics


def calculate_metrics(gen, val_loader, epoch, folder):
    # Initialize the lists to store the metric values
    psnr_values_vis = []
    ssim_values_vis = []
    # ff_values = []
    nmi_values_vis = []

    psnr_values_ir = []
    ssim_values_ir = []
    ff_values = []
    nmi_values_ir = []

    for batch in val_loader:
        img1 = batch['image_vis']
        img2 = batch['image_ir']
        a = batch['target_vis']
        b = batch['target_ir']
        # masked_feat = mask_feat(x, y)
        # mask_x, mask_y = masked_feat[2].to(config.DEVICE), masked_feat[3].to(config.DEVICE)
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        gen.eval()
        with torch.no_grad():
            # Perform image fusion
            y_fake, x_a, y_a, l_a = gen(img1, img2)
            # Calculate the metrics for Vis
            psnr_vis = PSNR(img1, y_fake)
            ssim_vis = SSIM(img1, y_fake)
            ff = FF(img1, img2, y_fake)
            nmi_vis = NMI(img1, y_fake)
            
            # ff = FF(img1, img2, y_fake)
            

            # Append the metric values to the lists
            psnr_values_vis.append(psnr_vis)
            ssim_values_vis.append(ssim_vis)
            ff_values.append(ff)
            nmi_values_vis.append(nmi_vis)

            # Calculate the metrics for IR

            psnr_ir = PSNR(img2, y_fake)
            ssim_ir = SSIM(img2, y_fake)
            ff = FF(img1, img2, y_fake)
            nmi_ir = NMI(img2, y_fake)

            # Append the metric values to the lists
            psnr_values_ir.append(psnr_ir)
            ssim_values_ir.append(ssim_ir)
            nmi_values_ir.append(nmi_ir)

    # # Create a dictionary to store the metric values
    metrics = {'PSNR_VIS': psnr_values_vis,
               'SSIM_VIS': ssim_values_vis,
               'NMI_VIS': nmi_values_vis,
               'FF': ff_values,
               'PSNR_IR' : psnr_values_ir,
               'SSIM_IR' : ssim_values_ir,
               'NMI_IR' : nmi_values_ir}

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(metrics)

    pd.options.display.float_format = '{:.2f}'.format
    # Calculate the statistics
    statistics = df.describe().apply(lambda s: s.apply('{0:.2f}'.format))
    # statistics = statistics.to_numpy().astype(float)

    plt.clf()

    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Create a table plot using the pandas built-in function
    table = plt.table(cellText=statistics.values,
                    colLabels=statistics.columns,
                    rowLabels = statistics.index,
                    loc='center')
    plt.title(f'Stats at epoch {epoch}')

    # Set the font size of the table
    # table.set_row_colors(['lightgray'] * len(statistics.index))
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Hide the axis
    plt.axis('off')

    # Save the figure as an image
    plt.savefig(os.path.join(folder, f'statistics_{epoch}.png'))
    plt.clf()
    # # Plot the metrics
    # fig, ax = plt.subplots()
    # for metric_name, metric_values in metrics.items():
    #     ax.plot(metric_values, label=metric_name)

    #     ax.set_title('Metrics')
    #     ax.set_xlabel('Epochs')
    #     ax.set_ylabel('Metric Value')
    #     ax.legend(title='Epoch', loc='upper left')
    #     ax.grid(True)
    #     plt.savefig(os.path.join(folder, f'metrics_{epoch}.png'))
    #     # plt.show()
    #     # plt.clf()

    # fig, axs = plt.subplots(2, 4, figsize=(12, 8), sharex = True)
    # fig.suptitle('Metrics', fontsize=16)
    # axs = axs.ravel()  # flatten the axes array for easy indexing

    # for i, (metric_name, metric_values) in enumerate(metrics.items()):
    #     # plot the metric on its own subplot
    #     axs[i].plot(metric_values)
    #     axs[i].set_title(metric_name)
    #     axs[i].set_xlabel('Epochs')
    #     axs[i].set_ylabel(metric_name)

    #     # add the epoch number as a legend
    #     axs[i].legend([f'epoch_{epoch}'], loc='upper left')

    # adjust the layout of the subplots and save the figure
    # fig.tight_layout()
    # plt.savefig(os.path.join(folder, f'metrics.png'))


    # Return the DataFrame and statistics
    return df, statistics