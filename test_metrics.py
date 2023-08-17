import cv2
from sewar.full_ref import psnr, ssim, mse, vifp, rmse, msssim, scc, uqi # currently not using rmse_sw because of dict size
from skimage.metrics import normalized_mutual_information as nmi
from csv import writer
from numpy import shape
from numpy.linalg import norm
# import plotly
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os

OURS = 0
PLOT = 1
'''
This script is a sandbox for testing image fusion metrics and will ultimately be used 
to quantitatively compare the performance of the different fusion methods and plot results.

Isaac TODO:
1. sewar: mse, psnr, ssim, vifp, rmse, msssim, scc, uqi (maybe), rmse_sw

2. skimage: NMI, mutual_info_score, structural_similarity, peak_signal_noise_ratio

3. Find or implement spatial frequency metric (SF)
SF: https://www.mathworks.com/matlabcentral/fileexchange/68753-spatial-frequency-sf

remove: VIFP, SF, SCC, RMSE, SSIM (good metric but MSSSIM does same thing and is more comprehensive)
keep: UQI instead of SCC (talk about why), MSE, PSNR, MSSSIM, NMI
'''


# def plot_data(filepath_ours:str="./eval_data_ad_Mfnet1.gpickle", filepath_theirs:str="./eval_data_Sea.gpickle"):
#   table_ours = pickle.load(open(filepath_ours, 'rb'))
#   table_theirs = pickle.load(open(filepath_theirs, 'rb'))

#   # keys_ours = list(table_ours.keys())
#   # keys_theirs = list(table_theirs.keys())
#   # axs = (axs1, axs2)

#   fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(8, 10))
#   # ax = plt.gca()
#   plt.rc('axes', titlesize=30)
#   plt.rc('xtick', labelsize=30) 
#   plt.rc('ytick', labelsize=30)
#   plt.rcParams['text.usetex'] = True
#   medianprops = dict(linestyle='-', linewidth=5, markeredgecolor='black',color='firebrick')
#   meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick', markersize=10)
#   boxprops = dict(linestyle='-', linewidth=6, color='darkgoldenrod')
#   whiskprops = dict(linestyle='-', linewidth=4)

#   key_list = ['NMI', 'PSNR']

#   offset = 0.18
#   pos = 0.2
#   ticks = []
#   tick_list = []

#   arrow_dict = {'NMI' : r'(${\uparrow}$ is better)', 'PSNR': r'(${\uparrow}$ is better)'}

#   for idx, metric in enumerate(key_list):
#     # plot the boxes for our metrics
#     ours_rgb_ir=[[val[0] for val in table_ours[metric]],[val[1] for val in table_ours[metric]]]
#     ours_box = axs[idx].boxplot(ours_rgb_ir, positions=[pos, pos+offset],capprops = whiskprops,whiskerprops = whiskprops,boxprops = boxprops, showmeans=True, meanprops=meanpointprops, medianprops=medianprops)
    
#     # add separating lines
#     x1 = pos - offset
#     axs[idx].vlines(x1, 0, 1, colors='silver', linestyles='solid', label='')
    
#     # add location for tick mark and label
#     ticks.append((pos + pos + offset)/2)
#     tick_list.append('MISFIT-V')
#     x2 = pos+offset + offset
    
#     # add metric label to top of this area of the plot
#     axs[idx].text(x2, max(ours_rgb_ir[1]) + 0.03, metric + ' ' + arrow_dict[metric], horizontalalignment='center', fontsize = 28, color='black', weight='bold')
    
#     # add a shaded area to separate the two sides
#     axs[idx].axvspan(x1, x2, ymin=0, ymax= 0.926, color='lightsteelblue', alpha=0.75, lw=0)
#     axs[idx].axvspan(x2, x2+(x2-x1), ymin=0, ymax= 0.926, color='navajowhite', alpha=0.75, lw=0)
    
#     # plot the boxes for their metrics
#     pos = pos + offset*3
#     theirs_rgb_ir=[[val[0] for val in table_theirs[metric]],[val[1] for val in table_theirs[metric]]]
#     theirs_box = axs[idx].boxplot(theirs_rgb_ir, positions=[pos,pos+offset ],capprops = whiskprops,whiskerprops = whiskprops,boxprops = boxprops, showmeans=True,  meanprops=meanpointprops,medianprops=medianprops)
#     ticks.append((pos + pos + offset)/2)
#     tick_list.append('SeAFusion')

#     # add a shaded area to separate the two sides
#     # plt.axvspan(x2, pos+offset + offset, color='antiquewhite', alpha=0.75, lw=0)
#     axs[idx].vlines(pos+offset + offset, 0, 1, colors='silver', linestyles='solid', label='')
#     pos = pos + offset*5

#       # set colors and labels for the boxes
#     ours_box["boxes"][0].set_color('blue')
#     ours_box["boxes"][1].set_color('red')
#     ours_box["boxes"][0].set_label('RGB')
#     ours_box["boxes"][1].set_label('IR')

#     theirs_box["boxes"][0].set_color('blue')
#     theirs_box["boxes"][1].set_color('red')

#   plt.show()


def plot_data_new(filepath_ours="./eval_data_ad_Mfnet1.gpickle", filepath_theirs="./eval_data_Mfnet_1L1.gpickle", filepath_theirs2="./eval_data__Mfnet_KL.gpickle"):
  table_ours = pickle.load(open(filepath_ours, 'rb'))
  table_theirs = pickle.load(open(filepath_theirs, 'rb'))
  table_theirs2 = pickle.load(open(filepath_theirs2, 'rb'))
  keys_ours = list(table_ours.keys())
  keys_theirs = list(table_theirs.keys())
  keys_theirs2 = list(table_theirs.keys())

  # set plot configs
  fig= plt.figure(figsize=(11, 9))
  ax = plt.gca()
  plt.rc('axes', titlesize=30)
  plt.rc('xtick', labelsize=30) 
  plt.rc('ytick', labelsize=30)
  plt.rcParams['text.usetex'] = True
  medianprops = dict(linestyle='-', linewidth=5, markeredgecolor='black',color='firebrick')
  meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick', markersize=10)
  boxprops = dict(linestyle='-', linewidth=6, color='darkgoldenrod')
  whiskprops = dict(linestyle='-', linewidth=4)
  # key_list = ['UQI', 'MSSSIM', 'MSE']
  key_list = ['NMI']
  # key_list = ["PSNR"]

  arrow_dict = {'NMI' : r'(${\uparrow}$ is better)', 'UQI': r'(${\uparrow}$ is better)', 'MSSSIM': r'(${\uparrow}$ is better)', 'MSE': r'(${\downarrow}$ is better)', 'PSNR': r'(${\uparrow}$ is better)'}

  offset = 0.25
  pos = 0.25
  ticks = []
  tick_list = []
  text_pos = []

  # for each metric in "key_list", we'll plot our results (RGB and IR) on one side, and theirs on the other
  for metric in key_list:
    # plot the boxes for our metrics
    ours_rgb_ir=[[val[0] for val in table_ours[metric]],[val[1] for val in table_ours[metric]]]
    ours_box = plt.boxplot(ours_rgb_ir, positions=[pos, pos+offset],capprops = whiskprops,whiskerprops = whiskprops,boxprops = boxprops, showmeans=True, meanprops=meanpointprops, medianprops=medianprops)
    
    # add separating lines
    x1 = pos - offset
    plt.vlines(x1, 1, 1.5, colors='silver', linestyles='solid', label='') # NMI
    # plt.vlines(x1, 0, 35, colors='silver', linestyles='solid', label='') #PSNR
    # plt.vlines(x1, 0, 1, colors='silver', linestyles='solid', label='') # other 3
    
    # add location for tick mark and label
    ticks.append((pos + pos + offset)/2)
    tick_list.append('Original')
    x2 = pos+offset + offset
    
    # add metric label to top of this area of the plot
    plt.text(1.5 * x2, 1.515, metric + ' ' + arrow_dict[metric], horizontalalignment='center', fontsize = 28, color='black', weight='bold') #NMI
    # plt.text(1.5 * x2, 35.8, metric + ' ' + arrow_dict[metric], horizontalalignment='center', fontsize = 28, color='black', weight='bold') # PSNR
    # plt.text(1.5*x2, 1.03, metric + ' ' + arrow_dict[metric], horizontalalignment='center', fontsize = 28, color='black', weight='bold') # other 3
    
    # add a shaded area to separate the two sides
    #NMI
    plt.axvspan(x1, x2, ymin=0, ymax= 0.91, color='lightsteelblue', alpha=0.75, lw=0)
    plt.axvspan(x2, x2+(x2-x1), ymin=0, ymax= 0.91, color='navajowhite', alpha=0.75, lw=0)
    plt.axvspan(x2+(x2-x1), x2+(x2-x1)+(x2-x1), ymin=0, ymax= 0.91, color='lightsteelblue', alpha=0.75, lw=0)

    #PSNR
    # plt.axvspan(x1, x2, ymin=0, ymax= 0.92, color='lightsteelblue', alpha=0.75, lw=0)
    # plt.axvspan(x2, x2+(x2-x1), ymin=0, ymax= 0.92, color='navajowhite', alpha=0.75, lw=0)
    # plt.axvspan(x2+(x2-x1), x2+(x2-x1)+(x2-x1), ymin=0, ymax= 0.92, color='lightsteelblue', alpha=0.75, lw=0)

    # Other 3
    # plt.axvspan(x1, x2, ymin=0, ymax= 0.926, color='lightsteelblue', alpha=0.75, lw=0)
    # plt.axvspan(x2, x2+(x2-x1), ymin=0, ymax= 0.926, color='navajowhite', alpha=0.75, lw=0)
    # plt.axvspan(x2+(x2-x1), x2+(x2-x1)+(x2-x1), ymin=0, ymax= 0.926, color='lightsteelblue', alpha=0.75, lw=0)

    # plot the boxes for their metrics
    pos = pos + offset*3
    theirs_rgb_ir=[[val[0] for val in table_theirs[metric]],[val[1] for val in table_theirs[metric]]]
    theirs_box = plt.boxplot(theirs_rgb_ir, positions=[pos,pos+offset ],capprops = whiskprops,whiskerprops = whiskprops,boxprops = boxprops, showmeans=True,  meanprops=meanpointprops,medianprops=medianprops)
    ticks.append((pos + pos + offset)/2)
    tick_list.append('$\lambda_\mathrm{L1}$ = 1')

    # add a shaded area to separate the two sides
    plt.vlines(pos+offset + offset, 1, 1.5, colors='silver', linestyles='solid', label='') # NMI
    # plt.vlines(pos+offset + offset, 0, 35, colors='silver', linestyles='solid', label='') #PSNR
    # plt.vlines(pos+offset + offset, 0, 1, colors='silver', linestyles='solid', label='') # other 3
    # pos = pos + offset*5

      # set colors and labels for the boxes
    ours_box["boxes"][0].set_color('blue')
    ours_box["boxes"][1].set_color('red')
    ours_box["boxes"][0].set_label('RGB')
    ours_box["boxes"][1].set_label('IR')

    theirs_box["boxes"][0].set_color('blue')
    theirs_box["boxes"][1].set_color('red')

    # plot the boxes for third case of no KL loss
    pos = pos + offset*3
    theirs2_rgb_ir=[[val[0] for val in table_theirs2[metric]],[val[1] for val in table_theirs2[metric]]]
    theirs2_box = plt.boxplot(theirs2_rgb_ir, positions=[pos,pos+offset ],capprops = whiskprops,whiskerprops = whiskprops,boxprops = boxprops, showmeans=True,  meanprops=meanpointprops,medianprops=medianprops)
    ticks.append((pos + pos + offset)/2)
    tick_list.append('No-KL')

    # add a shaded area to separate the two sides
    plt.vlines(pos+offset + offset, 1, 1.5, colors='silver', linestyles='solid', label='') # NMI
    # plt.vlines(pos+offset + offset, 0, 35, colors='silver', linestyles='solid', label='') #PSNR
    # plt.vlines(pos+offset + offset, 0, 1, colors='silver', linestyles='solid', label='') # other 3
    # pos = pos + offset*5

      # set colors and labels for the boxes
    ours_box["boxes"][0].set_color('blue')
    ours_box["boxes"][1].set_color('red')
    ours_box["boxes"][0].set_label('RGB')
    ours_box["boxes"][1].set_label('IR')

    # theirs_box["boxes"][0].set_color('blue')
    # theirs_box["boxes"][1].set_color('red')

    theirs2_box["boxes"][0].set_color('blue')
    theirs2_box["boxes"][1].set_color('red')

  # add final touches to plot
  plt.grid(axis='y', linestyle='--')
  plt.xticks(ticks, tick_list, fontsize=25)
  plt.yticks([1, 1.1, 1.2, 1.3, 1.4, 1.5], fontsize=25) # NMI
  # plt.yticks([0, 5, 10, 15, 20, 25, 30, 35], fontsize=25) # PSNR
  # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=25) # the other 3

  # title = 'Quantitative Comparison (uparrow is better))'
  # ax.set_title(title, fontsize=22, weight='bold')
  plt.title('Quantitative Comparison')
  # plt.title("Stuff")
  # trick: add lines in order to make the legend we want, then turn the line visibility off
  hB, = plt.plot([1,1],'b-')
  hR, = plt.plot([1,1],'r-')
  plt.legend((hB, hR),('RGB', 'IR'), fontsize=25)
  hB.set_visible(False)
  hR.set_visible(False)
  ax.set_ylim([1,1.55]) # NMI
  # ax.set_ylim([0, 38]) #PSNR
  # ax.set_ylim([0, 1.08]) # other 3
  # ax.set_xlim([-0.1, 1.2]) #NMI, PSNR
  plt.show() 



if __name__ == '__main__':
  # image format
  # test_images
  # -- 0_IR.png
  # -- 0_VIS.png
  # -- 0_Fused.png
  # -- 1_IR.png
  # -- 1_VIS.png
  # -- 1_Fused.png

  # Dict of evaluation functions and their metric description to iterate through
  funcs = {mse: "MSE", nmi: "NMI", psnr: "PSNR", msssim: "MSSSIM", uqi: "UQI"}

  dir = "./dump11/"

  dic = {"MSE": [], "NMI": [], "PSNR": [], "MSSSIM": [], "UQI": []}
  
  l = list(dic.keys())

  base_dir = ""
  if OURS == 1:
    base_dir = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/SeAFusion/Aad_Mfnet1/"
  else:
    base_dir = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/SeAFusion/Aad_Mfnet_1L1/"
  
  # base_v_dir = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/val/img/"
  # base_i_dir = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/val/img/"
  
  base_v_dir = "/mnt/mass_storage/ir_det_dataset/test/rgb/"
  base_i_dir = "/mnt/mass_storage/ir_det_dataset/test/fir/"

  if PLOT == 0:
    visual_files = sorted(os.listdir(base_v_dir))
    thermal_files = sorted(os.listdir(base_i_dir))
    fused_files = sorted(os.listdir(base_dir))

    idx = 0

    for vis_file, ir_file, f_file in zip(visual_files, thermal_files, fused_files):

      # if idx == 20:
      #   break

      fused_img = cv2.imread(''.join([base_dir, f_file]), cv2.IMREAD_GRAYSCALE)
      vis_img = cv2.imread(''.join([base_v_dir, vis_file]), cv2.IMREAD_GRAYSCALE)
      ir_img = cv2.imread(''.join([base_i_dir, ir_file]), cv2.IMREAD_GRAYSCALE)  

      vis_w = vis_img.shape[0]
      vis_h = vis_img.shape[1]
      left = int((vis_w - (45/64) * vis_w) / 2)
      right = int((vis_w + (45/64) * vis_w) / 2)
      vis_img = vis_img[0:vis_h, left:right]
      # vis_img = cv2.resize(vis_img, (1024, 512))
      # ir_img = cv2.resize(ir_img, (1024, 512))

      vis_img = cv2.resize(vis_img, (512, 256))
      ir_img = cv2.resize(ir_img, (512, 256))

      # Iterate over evaluation functions
      for func in list(funcs.keys()):
        if func != vifp:
          if func == mse:
            # pass
            temp_vis_img = cv2.normalize(vis_img, None, 0, 1.0, cv2.NORM_MINMAX)
            temp_ir_img = cv2.normalize(ir_img, None, 0, 1.0, cv2.NORM_MINMAX)
            temp_fused_img = cv2.normalize(fused_img, None, 0, 1.0, cv2.NORM_MINMAX)
            dic[funcs[func]].append([func(temp_vis_img, temp_fused_img), func(temp_ir_img, temp_fused_img)])
          else:
            dic[funcs[func]].append([func(vis_img, fused_img), func(ir_img, fused_img)])
        elif func == vifp:
          dic[funcs[func]].append([func(vis_img, fused_img)])
        else:
          dic[funcs[func]].append([func(fused_img)])
      
      idx += 1

    with open('./eval_data_' + base_dir[-10:-1] + '.gpickle', 'wb') as save_file:
      pickle.dump(dic, save_file, pickle.HIGHEST_PROTOCOL)
  
  else:
    plot_data_new()