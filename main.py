import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, save_generated_image
import torch.nn as nn
import torch.optim as optim
# import config
# from dataset import MapDataset
from Dataset import CustomDataSet
from GenAttn import Generator_attn
from GenAttn import Gen

from Discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from ssim import SSIM
from Metrics import calculate_metrics
import matplotlib.pyplot as plt 
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True


import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to use for training (default: cuda if available else cpu)")
parser.add_argument('--train_dir_vis', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/train/img',
                    help = "Path to the training directory for visual modality images")
parser.add_argument('--val_dir_vis', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/val/img',
                    help = "Path to the validation directory for visual modality images")
parser.add_argument('--train_dir_ir', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/train/img',
                    help = "Path to the training directory for thermal modality images")
parser.add_argument('--val_dir_ir', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/val/img',
                    help = "Path to the validation directory for thermal modality images")
parser.add_argument('--train_dir_vis_lbl', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/train/labels',
                    help = "Path to the training directory for visual modality images labels")
parser.add_argument('--val_dir_vis_lbl', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/val/label',
                    help = "Path to the validation directory for visual modality images labels")
parser.add_argument('--train_dir_ir_lbl', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/train/labels',
                    help = "Path to the training directory for thermal modality images labels")
parser.add_argument('--val_dir_ir_lbl', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed//DataLabels/IR/val/labels',
                    help = "Path to the validation directory for thermal modality images labels")
parser.add_argument("--learning_rate", type=float, default=2e-4,
                    help="Learning rate for the optimizer (default: 2e-4)")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for the training (default: 8)")
parser.add_argument("--num_workers", type=int, default=2,
                    help="Number of workers for the data loader (default: 2)")
parser.add_argument("--image_size", type=int, default=512,
                    help="Image size (default: 512)")
parser.add_argument("--channels_img", type=int, default=3,
                    help="Number of channels in the input images (default: 3)")
parser.add_argument("--l1_lambda", type=float, default=100,
                    help="Weight for L1 loss (default: 100)")
parser.add_argument("--alpha", type=float, default=5,
                    help="Weight for the adversarial loss for the generator (default: 5)")
parser.add_argument("--beta", type=float, default=10,
                    help="Weight for the identity loss for the generator (default: 10)")
parser.add_argument("--num_epochs", type=int, default=12,
                    help="Number of epochs for training (default: 12)")
parser.add_argument("--load_model", action="store_true",
                    help="Load a saved model for training (default: False)")
parser.add_argument("--save_model", action="store_true",
                    help="Save the trained model (default: True)")
parser.add_argument("--checkpoint_disc_ir", type=str, default="disc_ir_maskbwself.pth.tar",
                    help="Checkpoint file for the discriminator for infrared modality (default: disc_ir_maskbwself.pth.tar)")
parser.add_argument("--checkpoint_disc_vis", type=str, default="disc_vis_maskbwself.pth.tar",
                    help="Checkpoint file for the discriminator for visual modality (default: disc_vis_maskbwself.pth.tar)")
parser.add_argument("--checkpoint_gen", type=str, default="gen_10_maskbself.pth.tar",
                    help="Checkpoint file for the generator (default: gen_10_maskbself.pth.tar)")

args = parser.parse_args()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
train_dir_vis = args.train_dir_vis
val_dir_vis = args.val_dir_vis
train_dir_ir = args.train_dir_ir
val_dir_ir = args.val_dir_ir

train_dir_vis_lbl = args.train_dir_vis_lbl
val_dir_vis_lbl = args.val_dir_vis_lbl
train_dir_ir_lbl = args.train_dir_ir_lbl
val_dir_ir_lbl = args.val_dir_ir_lbl

learning_rate = args.learning_rate
batch_size = args.batch_size
num_workers = args.num_workers
image_size = args.image_size
channels_img = args.channels_img
l1_lambda = args.l1_lambda
alpha = args.alpha
beta = args.beta
num_epochs = args.num_epochs
load_model = args.load_model
save_model = args.save_model
checkpoint_disc_ir = args.checkpoint_disc_ir
checkpoint_disc_vis = args.checkpoint_disc_vis
checkpoint_gen = args.checkpoint_gen














def train_fn(
    disc_ir, disc_vis, gen, loader, opt_disc_ir, opt_disc_vis, opt_gen, l1_loss, bce, ssim, KL, g_scaler, d_scaler_ir,d_scaler_vis,
):
    loop = tqdm(loader, leave=True)

    for idx, batch in enumerate(loop):
    # for batch in loader:
        x = batch['image_vis']
        y = batch['image_ir']
        a = batch['target_vis']
        b = batch['target_ir']

        x = x.to(DEVICE)
        y = y.to(DEVICE)


        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake, x_a, y_a = gen(x, y)
            to_PIL = transforms.ToPILImage()
            D_real_ir = disc_ir(y, y)
            D_real_loss_ir = bce(D_real_ir, torch.ones_like(D_real_ir))
            D_fake_ir = disc_ir(y, y_fake.detach())
            D_fake_loss_ir = bce(D_fake_ir, torch.zeros_like(D_fake_ir))
            D_loss_ir = (D_real_loss_ir + D_fake_loss_ir) / 2 

            D_real_vis = disc_vis(x, x)
            D_real_loss_vis = bce(D_real_vis, torch.ones_like(D_real_vis))
            D_fake_vis = disc_vis(x, y_fake.detach())
            D_fake_loss_vis = bce(D_fake_vis, torch.zeros_like(D_fake_vis))
            D_loss_vis = (D_real_loss_vis + D_fake_loss_vis) / 2 
            
            # weightage preference for discriminator based on which modality has more number of humans
            # if torch.sum(a) < torch.sum(b):
            #     D_loss_ir = 5 * D_loss_ir
            # if torch.sum(a) > torch.sum(b):
            #     D_loss_vis = 5 * D_loss_vis  

        disc_ir.zero_grad()
        disc_vis.zero_grad()
        d_scaler_ir.scale(D_loss_ir).backward()
        d_scaler_vis.scale(D_loss_vis).backward()
        d_scaler_ir.step(opt_disc_ir)
        d_scaler_vis.step(opt_disc_vis)
        d_scaler_ir.update()
        d_scaler_vis.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake_ir = disc_ir(y, y_fake)
            D_fake_vis = disc_vis(x, y_fake)
            # print(torch.sum(a))
            # print(torch.sum(b))
            # print(max(torch.sum(b),torch.sum(a)))
            # detection_loss = abs(num_det - max(torch.sum(a), torch.sum(b)))
            G_fake_loss_ir = bce(D_fake_ir, torch.ones_like(D_fake_ir))
            G_fake_loss_vis = bce(D_fake_vis, torch.ones_like(D_fake_vis))
            # L1 = (l1_loss(y_fake, y) + l1_loss(y_fake, x) - l1_loss(x_a.to(config.DEVICE), y_a.to(config.DEVICE))) * config.L1_LAMBDA
            L1 = (l1_loss(y_fake, y) + l1_loss(y_fake, x)) * l1_lambda
            KL1 = -KL(x_a.clone(), y_a.clone())
            G_loss = G_fake_loss_ir + G_fake_loss_vis + KL1 + L1 + beta * (KL(y_fake.clone(),y.clone()) + KL(y_fake.clone(),x.clone()))

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real_ir=torch.sigmoid(D_real_ir).mean().item(),
                D_fake_ir=torch.sigmoid(D_fake_ir).mean().item(),
                D_real_vis=torch.sigmoid(D_real_vis).mean().item(),
                D_fake_vis=torch.sigmoid(D_fake_vis).mean().item(),
                G_Loss = torch.tensor(G_loss).clone().detach().mean().item(),
            )


def main():
    # masked_feat = MaskedFeatures(in_chan = 3, features = 8)
    disc_ir = Discriminator(in_channels=3).to(DEVICE)
    disc_vis = Discriminator(in_channels=3).to(DEVICE)
    # gen = Generator(in_channels=64, features=64).to(config.DEVICE)
    # gen = Generator_attn(3,32).to(config.DEVICE)
    gen = Gen().to(DEVICE)
    opt_disc_ir = optim.Adam(disc_ir.parameters(), lr=learning_rate, betas=(0.5, 0.999),)
    opt_disc_vis = optim.Adam(disc_vis.parameters(), lr=learning_rate, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    transform = transforms.Compose([transforms.Resize((512,1024),transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    # transform = transforms.Compose([transforms.Resize((512,512),transforms.InterpolationMode.BILINEAR)])
    # transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((512,512),transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    ssim = SSIM()
    KL = nn.KLDivLoss()

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    #     )

    train_dataset = CustomDataSet(train_dir_vis, train_dir_ir, train_dir_vis_lbl, train_dir_ir_lbl, transform= transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler_ir = torch.cuda.amp.GradScaler()
    d_scaler_vis = torch.cuda.amp.GradScaler()
    val_dataset = CustomDataSet(val_dir_vis, val_dir_ir, val_dir_vis_lbl, val_dir_ir_lbl, transform)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)
    val_loader1 = DataLoader(val_dataset, batch_size = 2, shuffle=False)

    min_psnr_vis = []
    max_psnr_vis = []
    mean_psnr_vis = []
    min_ssim_vis = []
    max_ssim_vis = []
    mean_ssim_vis = []
    min_nmi_vis = []
    max_nmi_vis = []
    mean_nmi_vis = []

    min_psnr_ir = []
    max_psnr_ir = []
    mean_psnr_ir = []
    min_ssim_ir = []
    max_ssim_ir = []
    mean_ssim_ir = []
    min_nmi_ir = []
    max_nmi_ir = []
    mean_nmi_ir = []

    for epoch in range(num_epochs):
        train_fn(
            disc_ir, disc_vis, gen, train_loader, opt_disc_ir, opt_disc_vis, opt_gen, L1_LOSS, BCE, ssim, KL, g_scaler, d_scaler_ir,d_scaler_vis
        )
        


        if save_model and epoch % 2 == 0:
            # save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            # save_checkpoint(disc_ir, opt_disc_ir, filename=config.CHECKPOINT_DISC_IR)
            # save_checkpoint(disc_vis, opt_disc_vis, filename=config.CHECKPOINT_DISC_VIS)
            print('plotting metrics')
            df, stats = calculate_metrics(gen, val_loader1, epoch, folder = "dump7")
            min_psnr_vis.append(stats.loc['min', 'PSNR_VIS'])
            max_psnr_vis.append(stats.loc['max', 'PSNR_VIS'])
            mean_psnr_vis.append(stats.loc['mean', 'PSNR_VIS'])
            min_psnr_ir.append(stats.loc['min', 'PSNR_IR'])
            max_psnr_ir.append(stats.loc['max', 'PSNR_IR'])
            mean_psnr_ir.append(stats.loc['mean', 'PSNR_IR'])

            min_ssim_vis.append(stats.loc['min', 'SSIM_VIS'])
            max_ssim_vis.append(stats.loc['max', 'SSIM_VIS'])
            mean_ssim_vis.append(stats.loc['mean', 'SSIM_VIS'])
            min_ssim_ir.append(stats.loc['min', 'SSIM_IR'])
            max_ssim_ir.append(stats.loc['max', 'SSIM_IR'])
            mean_ssim_ir.append(stats.loc['mean', 'SSIM_IR'])

            min_nmi_vis.append(stats.loc['min', 'NMI_VIS'])
            max_nmi_vis.append(stats.loc['max', 'NMI_VIS'])
            mean_nmi_vis.append(stats.loc['mean', 'NMI_VIS'])
            min_nmi_ir.append(stats.loc['min', 'NMI_IR'])
            max_nmi_ir.append(stats.loc['max', 'NMI_IR'])
            mean_nmi_ir.append(stats.loc['mean', 'NMI_IR'])

            print(f"saving examples at epoch {epoch}")
            save_some_examples(gen, val_loader, epoch, folder="dump6", device = DEVICE)  #label3 512x1024 with no clamp on masks and yolov5

    # plot min and max lists in separate subplots
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # plot for min and max psnr_vis
    fig, axs = plt.subplots(1,6, figsize = (15,5))
    axs[0].plot(min_psnr_vis, label='min PSNR_VIS')
    axs[0].plot(max_psnr_vis, label='max PSNR_VIS')
    axs[0].plot(mean_psnr_vis, label='mean PSNR_VIS')
    axs[0].set_title('PSNR_VIS')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('PSNR_VIS')
    axs[0].legend()

    # plot for min and max ssim_vis
    axs[1].plot(min_ssim_vis, label='min SSIM_VIS')
    axs[1].plot(max_ssim_vis, label='max SSIM_VIS')
    axs[1].plot(mean_ssim_vis, label='mean SSIM_VIS')
    axs[1].set_title('SSIM_VIS')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('SSIM_VIS')
    axs[1].legend()

    axs[2].plot(min_nmi_vis, label='min NMI_VIS')
    axs[2].plot(max_nmi_vis, label='max NMI_VIS')
    axs[2].plot(mean_nmi_vis, label='mean NMI_VIS')
    axs[2].set_title('NMI_VIS')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('NMI_VIS')
    axs[2].legend()

    axs[3].plot(min_psnr_ir, label='min PSNR_IR')
    axs[3].plot(max_psnr_ir, label='max PSNR_IR')
    axs[3].plot(mean_psnr_ir, label='mean PSNR_IR')
    axs[3].set_title('PSNR_IR')
    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('PSNR_IR')
    axs[3].legend()

    # plot for min and max ssim_vis
    axs[4].plot(min_ssim_ir, label='min SSIM_IR')
    axs[4].plot(max_ssim_ir, label='max SSIM_IR')
    axs[4].plot(mean_ssim_ir, label='mean SSIM_IR')
    axs[4].set_title('SSIM_IR')
    axs[4].set_xlabel('Epochs')
    axs[4].set_ylabel('SSIM_IR')
    axs[4].legend()

    axs[5].plot(min_nmi_ir, label='min NMI_IR')
    axs[5].plot(max_nmi_ir, label='max NMI_IR')
    axs[5].plot(mean_nmi_ir, label='mean NMI_IR')
    axs[5].set_title('NMI_IR')
    axs[5].set_xlabel('Epochs')
    axs[5].set_ylabel('NMI_IR')
    axs[5].legend()
    
    
    # adjust the layout of the subplots and save the figure
    fig.tight_layout()
    plt.savefig('min_max_metrics_visir.png')
            



if __name__ == "__main__":
    main()
