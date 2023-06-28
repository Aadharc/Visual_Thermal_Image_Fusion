import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, save_generated_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb

# import config
# from dataset import MapDataset
from Dataset2 import CustomDataSet
from GenAttn import Generator_attn, Gen

from torch.optim.lr_scheduler import CosineAnnealingLR

from Discriminator import Discriminator
# from Discriminator_attn import Discriminator_attn
from torch.utils.data import DataLoader, Subset, RandomSampler
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from ssim import SSIM
from loss import calculate_ssim
from Metrics import calculate_metrics
import matplotlib.pyplot as plt 
import argparse
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True




# Initialize wandb project
wandb.init(project = "Fusion")

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to use for training (default: cuda if available else cpu)")
parser.add_argument('--train_dir_vis', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/train/img',
                    help = "Path to the training directory for visual modality images")
parser.add_argument('--test_dir_vis', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/val/img',
                    help = "Path to the test data directory for visual modality images")
parser.add_argument('--train_dir_ir', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/train/img',
                    help = "Path to the training directory for thermal modality images")
parser.add_argument('--test_dir_ir', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/val/img',
                    help = "Path to the test data directory for thermal modality images")
parser.add_argument('--train_dir_vis_lbl', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/train/labels',
                    help = "Path to the training directory for visual modality images labels")
parser.add_argument('--test_dir_vis_lbl', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/val/label',
                    help = "Path to the test data directory for visual modality images labels")
parser.add_argument('--train_dir_ir_lbl', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/train/labels',
                    help = "Path to the training directory for thermal modality images labels")
parser.add_argument('--test_dir_ir_lbl', type=str, default='/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed//DataLabels/IR/val/labels',
                    help = "Path to the test data directory for thermal modality images labels")
parser.add_argument("--learning_rate", type=float, default=2e-4,
                    help="Learning rate for the optimizer (default: 2e-4)")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for the training (default: 8)")
parser.add_argument("--num_workers", type=int, default=2,
                    help="Number of workers for the data loader (default: 2)")
parser.add_argument("--image_size", type=int, default=(512, 1024),
                    help="Image size (default: (512, 1024)")
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
parser.add_argument("--load_model", action="store_true", default = False,
                    help="Load a saved model for training (default: False)")
parser.add_argument("--save_model", action="store_true", default = True,
                    help="Save the trained model (default: True)")
parser.add_argument("--checkpoint_disc_ir", type=str, default="disc_ir_maskbwself.pth.tar",
                    help="Checkpoint file for the discriminator for infrared modality (default: disc_ir_maskbwself.pth.tar)")
parser.add_argument("--checkpoint_disc_vis", type=str, default="disc_vis_maskbwself.pth.tar",
                    help="Checkpoint file for the discriminator for visual modality (default: disc_vis_maskbwself.pth.tar)")
parser.add_argument("--checkpoint_gen", type=str, default="gen_Danny.pth.tar",
                    help="Checkpoint file for the generator (default: gen_Danny.pth.tar)")

args = parser.parse_args()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
train_dir_vis = args.train_dir_vis
test_dir_vis = args.test_dir_vis
train_dir_ir = args.train_dir_ir
test_dir_ir = args.test_dir_ir

train_dir_vis_lbl = args.train_dir_vis_lbl
test_dir_vis_lbl = args.test_dir_vis_lbl
train_dir_ir_lbl = args.train_dir_ir_lbl
test_dir_ir_lbl = args.test_dir_ir_lbl

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

# saving config to wandb
wandb.config.learning_rate = learning_rate
wandb.config.batch_size = batch_size
wandb.config.num_workers = num_workers
wandb.config.image_size = image_size
channels_img = channels_img
wandb.config.l1_lambda = l1_lambda
wandb.config.alpha = alpha
wandb.config.beta = beta
wandb.config.num_epochs = num_epochs


def train_fn(disc_ir, disc_vis, gen, train_loader, val_loader, opt_disc_ir, opt_disc_vis, opt_gen, l1_loss, bce, ssim, KL, g_scaler, d_scaler_ir,d_scaler_vis):

    train_loop = tqdm(train_loader, leave=True)

    D_loss_ir_train = 0
    D_loss_vis_train = 0
    G_loss_train = 0

    gen.train()
    disc_ir.train()
    disc_vis.train()

    for idx, batch in enumerate(train_loop):
        # Training
        x = batch['image_vis']
        y = batch['image_ir']
        # a = batch['target_vis']
        # b = batch['target_ir']

        x = x.to(DEVICE)
        y = y.to(DEVICE)


        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake, attn1, attn2, l_a = gen(x, y)
            x_a = x * attn1
            y_a = y * attn2
            to_PIL = transforms.ToPILImage()
            D_real_ir = disc_ir(y, y)
            D_real_loss_ir = bce(D_real_ir, torch.ones_like(D_real_ir))
            D_fake_ir = disc_ir(y, y_fake.detach())
            D_fake_loss_ir = bce(D_fake_ir, torch.zeros_like(D_fake_ir))
            D_loss_ir = (D_real_loss_ir + D_fake_loss_ir) / 2 
            D_loss_ir_train += D_loss_ir.item()

            D_real_vis = disc_vis(x, x)
            D_real_loss_vis = bce(D_real_vis, torch.ones_like(D_real_vis))
            D_fake_vis = disc_vis(x, y_fake.detach())
            D_fake_loss_vis = bce(D_fake_vis, torch.zeros_like(D_fake_vis))
            D_loss_vis = (D_real_loss_vis + D_fake_loss_vis) / 2
            D_loss_vis_train += D_loss_vis.item() 
            
            # weightage preference for discriminator based on which modality has more number of humans
            # if torch.sum(a) < torch.sum(b):
            #     D_loss_ir = 50 * D_loss_ir
            # if torch.sum(a) > torch.sum(b):
            #     D_loss_vis = 50 * D_loss_vis  

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
            # triplet_loss = F.triplet_margin_loss(y_fake, y_a, x_a)
            cross = nn.CrossEntropyLoss()
            # attn_loss = (cross(y_fake * attn2, y * attn2) + cross(y_fake * attn1, x * attn1)) * 10
            attn_contrastive_loss = l_a
            # KL1 = KL(x_a.clone(), y_a.clone())
            # G_loss = G_fake_loss_ir + G_fake_loss_vis + L1 + attn_contrastive_loss
            G_loss = G_fake_loss_ir + G_fake_loss_vis + L1 + beta * (KL(y_fake.clone(),y.clone()) + KL(y_fake.clone(),x.clone())) + attn_contrastive_loss
            G_loss_train += G_loss.item()
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        G_loss_train /= len(train_loader)
        D_loss_ir_train /= len(train_loader)
        D_loss_vis_train /= len(train_loader)

        if idx % 10 == 0:

            train_loop.set_postfix(
                D_real_ir=torch.sigmoid(D_real_ir).mean().item(),
                D_fake_ir=torch.sigmoid(D_fake_ir).mean().item(),
                D_real_vis=torch.sigmoid(D_real_vis).mean().item(),
                D_fake_vis=torch.sigmoid(D_fake_vis).mean().item(),
                # G_Loss = torch.tensor(G_loss).clone().detach().mean().item(),
            )

    gen.eval()
    disc_ir.eval()
    disc_vis.eval()
    val_loop = tqdm(val_loader, leave=True)
    with torch.no_grad():
        for idx, batch in enumerate(val_loop):
            # Validation
            x = batch['image_vis']
            y = batch['image_ir']
            # a = batch['target_vis']
            # b = batch['target_ir']

            x = x.to(DEVICE)
            y = y.to(DEVICE)


            # Discriminator
            with torch.cuda.amp.autocast():
                y_fake, attn1, attn2, l_a = gen(x, y)
                x_a = x * attn1
                y_a = y * attn2
                to_PIL = transforms.ToPILImage()
                D_real_ir_val = disc_ir(y, y)
                D_real_loss_ir = bce(D_real_ir, torch.ones_like(D_real_ir))
                D_fake_ir_val = disc_ir(y, y_fake.detach())
                D_fake_loss_ir = bce(D_fake_ir, torch.zeros_like(D_fake_ir))
                D_loss_ir = (D_real_loss_ir + D_fake_loss_ir) / 2 

                D_real_vis_val = disc_vis(x, x)
                D_real_loss_vis = bce(D_real_vis, torch.ones_like(D_real_vis))
                D_fake_vis_val = disc_vis(x, y_fake.detach())
                D_fake_loss_vis = bce(D_fake_vis, torch.zeros_like(D_fake_vis))
                D_loss_vis = (D_real_loss_vis + D_fake_loss_vis) / 2 
                
                # weightage preference for discriminator based on which modality has more number of humans
                # if torch.sum(a) < torch.sum(b):
                #     D_loss_ir = 5 * D_loss_ir
                # if torch.sum(a) > torch.sum(b):
                #     D_loss_vis = 5 * D_loss_vis  

            # generator
            with torch.cuda.amp.autocast():
                D_fake_ir = disc_ir(y, y_fake)
                D_fake_vis = disc_vis(x, y_fake)

                G_fake_loss_ir = bce(D_fake_ir, torch.ones_like(D_fake_ir))
                G_fake_loss_vis = bce(D_fake_vis, torch.ones_like(D_fake_vis))
                # L1 = (l1_loss(y_fake, y) + l1_loss(y_fake, x) - l1_loss(x_a.to(config.DEVICE), y_a.to(config.DEVICE))) * config.L1_LAMBDA
                L1 = (l1_loss(y_fake, y) + l1_loss(y_fake, x)) * l1_lambda
                cross = nn.CrossEntropyLoss()
                # attn_loss = (cross(y_fake * attn2, y * attn2) + cross(y_fake * attn1, x * attn1)) * 10
                attn_contrastive_loss = l_a
                # triplet_loss = F.triplet_margin_loss(y_fake, y_a, x_a)
                # KL1 = KL(x_a.clone(), y_a.clone())
                # G_loss = G_fake_loss_ir + G_fake_loss_vis + L1 + attn_contrastive_loss
                G_loss = G_fake_loss_ir + G_fake_loss_vis + L1 + beta * (KL(y_fake.clone(),y.clone()) + KL(y_fake.clone(),x.clone())) + attn_contrastive_loss
                
            if idx % 10 == 0:
                val_loop.set_postfix(
                    D_real_ir_val=torch.sigmoid(D_real_ir).mean().item(),
                    D_fake_ir_val=torch.sigmoid(D_fake_ir).mean().item(),
                    D_real_vis_val=torch.sigmoid(D_real_vis).mean().item(),
                    D_fake_vis_val=torch.sigmoid(D_fake_vis).mean().item(),
                    G_Loss_val = torch.tensor(G_loss).clone().detach().mean().item(),
                )
            # Logging loss values for training
            # wandb.log({"Generator Loss Val": G_loss_val.item(), "Discriminator IR Loss Val": D_loss_ir.item(), "Discriminator VIS Loss Val": D_loss_vis.item()})
            

    return G_loss_train, D_loss_vis_train, D_loss_ir_train

def main():
    # masked_feat = MaskedFeatures(in_chan = 3, features = 8)
    disc_ir = Discriminator(in_channels=3).to(DEVICE)
    disc_vis = Discriminator(in_channels=3).to(DEVICE)
    # Log model and gradients
    wandb.watch(disc_ir)
    wandb.watch(disc_vis)
    # gen = Generator(in_channels=64, features=64).to(config.DEVICE)
    # gen = Generator_attn(3,32).to(DEVICE)
    # Log model and gradients
    gen = Gen().to(DEVICE)
    wandb.watch(gen)
    opt_disc_ir = optim.Adam(disc_ir.parameters(), lr=learning_rate, betas=(0.9, 0.999),)
    opt_disc_vis = optim.Adam(disc_vis.parameters(), lr=learning_rate, betas=(0.9, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(opt_gen,
                              T_max = 32, # Maximum number of iterations.
                             eta_min = 1e-4) # Minimum learning rate.
    transform = transforms.Compose([transforms.Resize((256,512),transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
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

    # train_dataset = CustomDataSet(train_dir_vis, train_dir_ir, train_dir_vis_lbl, train_dir_ir_lbl, transform= transform)
    # train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)
    # Define your dataset as before
    # dataset = CustomDataSet(train_dir_vis, train_dir_ir, train_dir_vis_lbl, train_dir_ir_lbl, transform=transform)
    dataset = CustomDataSet(train_dir_vis, train_dir_ir, transform=transform)

    # Calculate the size of the training set and validation set
    train_size = int(len(dataset) * 0.8)  # 80% for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation

    # Create random samplers for training and validation sets
    train_sampler = RandomSampler(dataset, num_samples=train_size, replacement=True)
    val_sampler = RandomSampler(dataset, num_samples=val_size, replacement=True)

    # Create data loaders for training and validation sets
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler_ir = torch.cuda.amp.GradScaler()
    d_scaler_vis = torch.cuda.amp.GradScaler()
    test_dataset = CustomDataSet(test_dir_vis, test_dir_ir,transform)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
    test_loader1 = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    # initializing lists to visualize metrics
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
        p,q,r = train_fn(
            disc_ir, disc_vis, gen, train_loader, val_loader, opt_disc_ir, opt_disc_vis, opt_gen, L1_LOSS, BCE, ssim, KL, g_scaler, d_scaler_ir,d_scaler_vis
        )
        
        tqdm.write(f"Epoch {epoch} lr {learning_rate} - Gen Loss: {p:.4f} - D1 Loss: {q:.4f} - D2 Loss: {r:.4f} ")
        # Logging loss values for training
        wandb.log({"Generator Loss": p, "Discriminator IR Loss": r, "Discriminator VIS Loss": q, "Learning Rate" : learning_rate})

        if save_model and epoch % 2 == 0:
            save_checkpoint(gen, opt_gen, filename= f"gen_Danny_{epoch}.pth.tar")
            # save_checkpoint(disc_ir, opt_disc_ir, filename=checkpoint_disc_ir)
            # save_checkpoint(disc_vis, opt_disc_vis, filename=checkpoint_disc_vis)
    #         print('plotting metrics')
    #         # m1 = calculate_metrics_with_patch(gen, test_loader1, epoch, folder = "dump7", device = DEVICE)
    #         df, stats = calculate_metrics(gen, test_loader1, epoch, folder = "dump19")
    #         min_psnr_vis.append(stats.loc['min', 'PSNR_VIS'])
    #         max_psnr_vis.append(stats.loc['max', 'PSNR_VIS'])
    #         mean_psnr_vis.append(stats.loc['mean', 'PSNR_VIS'])
    #         min_psnr_ir.append(stats.loc['min', 'PSNR_IR'])
    #         max_psnr_ir.append(stats.loc['max', 'PSNR_IR'])
    #         mean_psnr_ir.append(stats.loc['mean', 'PSNR_IR'])

    #         min_ssim_vis.append(stats.loc['min', 'SSIM_VIS'])
    #         max_ssim_vis.append(stats.loc['max', 'SSIM_VIS'])
    #         mean_ssim_vis.append(stats.loc['mean', 'SSIM_VIS'])
    #         min_ssim_ir.append(stats.loc['min', 'SSIM_IR'])
    #         max_ssim_ir.append(stats.loc['max', 'SSIM_IR'])
    #         mean_ssim_ir.append(stats.loc['mean', 'SSIM_IR'])

    #         min_nmi_vis.append(stats.loc['min', 'NMI_VIS'])
    #         max_nmi_vis.append(stats.loc['max', 'NMI_VIS'])
    #         mean_nmi_vis.append(stats.loc['mean', 'NMI_VIS'])
    #         min_nmi_ir.append(stats.loc['min', 'NMI_IR'])
    #         max_nmi_ir.append(stats.loc['max', 'NMI_IR'])
    #         mean_nmi_ir.append(stats.loc['mean', 'NMI_IR'])

            print(f"saving examples at epoch {epoch}")
            save_some_examples(gen, test_loader, epoch, folder="dump19", device = DEVICE)  #label3 512x1024 with no clamp on masks and yolov5         


    # # converting all data lists to np arrays to calculate mean for box plots
    # min_psnr_vis = np.array(min_psnr_vis , dtype = float)
    # max_psnr_vis = np.array(max_psnr_vis , dtype = float)
    # mean_psnr_vis = np.array( mean_psnr_vis, dtype = float)
    # min_ssim_vis = np.array(min_ssim_vis , dtype = float)
    # max_ssim_vis = np.array(max_ssim_vis , dtype = float)
    # mean_ssim_vis = np.array(mean_ssim_vis , dtype = float)
    # min_nmi_vis = np.array(min_nmi_vis , dtype = float)
    # max_nmi_vis = np.array(max_nmi_vis , dtype = float)
    # mean_nmi_vis = np.array(mean_nmi_vis , dtype = float)

    # min_psnr_ir = np.array(min_psnr_ir , dtype = float)
    # max_psnr_ir = np.array(max_psnr_ir , dtype = float)
    # mean_psnr_ir = np.array(mean_psnr_ir , dtype = float)
    # min_ssim_ir = np.array( min_ssim_ir, dtype = float)
    # max_ssim_ir = np.array(max_ssim_ir , dtype = float)
    # mean_ssim_ir = np.array( mean_ssim_ir , dtype = float)
    # min_nmi_ir = np.array( min_nmi_ir, dtype = float)
    # max_nmi_ir = np.array( max_nmi_ir, dtype = float)
    # mean_nmi_ir = np.array(mean_nmi_ir , dtype = float)

    # # Create subplots
    # fig, axs = plt.subplots(1, 2)

    # # Plot for IR Images
    # axs[0,0].plot(min_ssim_ir, label='Min SSIM')
    # axs[0,0].plot(max_ssim_ir, label='Max SSIM')
    # axs[0,0].plot(mean_ssim_ir, label='Mean SSIM')
    # axs[0,0].set_ylabel('SSIM Values (IR Images)')
    # axs[0,0].set_ylim(0, 1)
    # axs[0,0].legend()

    # # Plot for Visual Images
    # axs[0,1].plot(min_ssim_vis, label='Min SSIM')
    # axs[0,1].plot(max_ssim_vis, label='Max SSIM')
    # axs[0,1].plot(mean_ssim_vis, label='Mean SSIM')
    # axs[0,1].set_xlabel('Epochs')
    # axs[0,1].set_ylabel('SSIM Values (Visual Images)')
    # axs[0,1].set_ylim(0, 1)
    # axs[0,1].legend()

    # # Adjust spacing between subplots
    # fig.tight_layout()
    # plt.savefig('dump20_' + 'min_max_metrics_visir_subplot_ssim_box.png')
    # plt.close()


    # # print(type(mean_psnr_vis))
    # box_data = [min_psnr_vis, max_psnr_vis, mean_psnr_vis, min_ssim_vis, max_ssim_vis, mean_ssim_vis, min_nmi_vis, max_nmi_vis, mean_nmi_vis, min_psnr_ir, max_psnr_ir, mean_psnr_ir, min_ssim_ir, max_ssim_ir, mean_ssim_ir, min_nmi_ir, max_nmi_ir, mean_nmi_ir]
    # labels = ['PSNR_VIS', 'max PSNR_VIS', 'mean PSNR_VIS', 'SSIM_VIS', 'max SSIM_VIS', 'mean SSIM_VIS', 'NMI_VIS', 'max NMI_VIS', 'mean NMI_VIS', 'PSNR_IR', 'max PSNR_IR', 'mean PSNR_IR', 'SSIM_IR', 'max SSIM_IR', 'mean SSIM_IR', 'NMI_IR', 'max NMI_IR', 'mean NMI_IR']
    # # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # for i in range(2):
    #     # fig, ax = plt.subplots(figsize=(5,5))
    #     fig, axs = plt.subplots(1, 2, figsize=(5, 10), sharey = True)
    #     # if i == 0:
    #     #     axs[0, i].plot(min_psnr_vis, label='min PSNR_VIS')
    #     #     axs[0, i].plot(max_psnr_vis, label='max PSNR_VIS')
    #     #     axs[0, i].plot(mean_psnr_vis, label='mean PSNR_VIS')
    #     #     axs[0, i].set_title('PSNR_VIS')
    #     #     axs[0, i].set_xlabel('Epochs')
    #     #     axs[0, i].set_ylabel('PSNR_VIS')
    #     #     axs[0, i].set_ylim(40, 100)
    #     #     # axs[0, i].set_ylim([min(min_psnr_vis + max_psnr_vis + mean_psnr_vis), max(min_psnr_vis + max_psnr_vis + mean_psnr_vis)])
    #     #     axs[0, i].legend()
    #         # axs[1,i].boxplot(box_data[i*3:(i+1)*3])
    #         # axs[1,i].set_title(labels[i*3])
    #         # axs[1,i].set_xticklabels(['min', 'max', 'mean'])
    #         # axs[1,i].set_ylabel(labels[i*3])
    #         # ax.set_ylim(0, 100)
            
    #     if i == 0:
    #         axs[0, i].plot(min_ssim_vis, label='min SSIM_VIS')
    #         axs[0, i].plot(max_ssim_vis, label='max SSIM_VIS')
    #         axs[0, i].plot(mean_ssim_vis, label='mean SSIM_VIS')
    #         axs[0, i].set_title('SSIM_VIS')
    #         axs[0, i].set_xlabel('Epochs')
    #         axs[0, i].set_ylabel('SSIM_VIS')
    #         axs[0, i].set_ylim(0, 1)
    #         axs[0, i].legend()
    #     # elif i == 2:
    #     #     axs[0, i].plot(min_nmi_vis, label='min NMI_VIS')
    #     #     axs[0, i].plot(max_nmi_vis, label='max NMI_VIS')
    #     #     axs[0, i].plot(mean_nmi_vis, label='mean NMI_VIS')
    #     #     axs[0, i].set_title('NMI_VIS')
    #     #     axs[0, i].set_xlabel('Epochs')
    #     #     axs[0, i].set_ylabel('NMI_VIS')
    #     #     axs[0, i].set_ylim(0.9, 1.8)
    #     #     # axs[0, i].set_ylim([min(min_nmi_vis + max_nmi_vis + mean_nmi_vis), max(min_nmi_vis + max_nmi_vis + mean_nmi_vis)])
    #     #     axs[0, i].legend()
    #     # elif i == 3:
    #     #     axs[0, i].plot(min_psnr_ir, label='min PSNR_IR')
    #     #     axs[0, i].plot(max_psnr_ir, label='max PSNR_IR')
    #     #     axs[0, i].plot(mean_psnr_ir, label='mean PSNR_IR')
    #     #     axs[0, i].set_title('PSNR_IR')
    #     #     axs[0, i].set_xlabel('Epochs')
    #     #     axs[0, i].set_ylabel('PSNR_IR')
    #     #     axs[0, i].set_ylim(40, 100)
    #     #     # axs[0, i].set_ylim([min(min_psnr_ir + max_psnr_ir + mean_psnr_ir), max(min_psnr_ir + max_psnr_ir + mean_psnr_ir)])
    #     #     axs[0, i].legend()
    #     elif i == 1:
    #         axs[0, i].plot(min_ssim_ir, label='min SSIM_IR')
    #         axs[0, i].plot(max_ssim_ir, label='max SSIM_IR')
    #         axs[0, i].plot(mean_ssim_ir, label='mean SSIM_IR')
    #         axs[0, i].set_title('SSIM_IR')
    #         axs[0, i].set_xlabel('Epochs')
    #         axs[0, i].set_ylabel('SSIM_IR')
    #         axs[0, i].set_ylim(0, 1)
    #         # axs[0, i].set_ylim([min(min_ssim_ir + max_ssim_ir + mean_ssim_ir), max(min_ssim_ir + max_ssim_ir + mean_ssim_ir)])
    #         axs[0, i].legend()
    #     # elif i == 5:
    #     #     axs[0, i].plot(min_nmi_ir, label='min NMI_IR')
    #     #     axs[0, i].plot(max_nmi_ir, label='max NMI_IR')
    #     #     axs[0, i].plot(mean_nmi_ir, label='mean NMI_IR')
    #     #     axs[0, i].set_title('NMI_IR')
    #     #     axs[0, i].set_xlabel('Epochs')
    #     #     axs[0, i].set_ylabel('NMI_IR')
    #     #     axs[0, i].set_ylim(0.9, 1.8)
    #     #     # axs[0, i].set_ylim([min(min_nmi_ir + max_nmi_ir + mean_nmi_ir), max(min_nmi_ir + max_nmi_ir + mean_nmi_ir)])
    #     #     axs[0, i].legend()
    #     # adjust the layout of the subplots and save the figure
    # fig.tight_layout()
    # plt.savefig('dump19_' + 'min_max_metrics_visir_subplot_ssim_box.png')
    # plt.close()

    # # box_data = [min_psnr_vis, max_psnr_vis, mean_psnr_vis, min_ssim_vis, max_ssim_vis, mean_ssim_vis, min_nmi_vis, max_nmi_vis, mean_nmi_vis, min_psnr_ir, max_psnr_ir, mean_psnr_ir, min_ssim_ir, max_ssim_ir, mean_ssim_ir, min_nmi_ir, max_nmi_ir, mean_nmi_ir]
    # # labels = ['min PSNR_VIS', 'max PSNR_VIS', 'mean PSNR_VIS', 'min SSIM_VIS', 'max SSIM_VIS', 'mean SSIM_VIS', 'min NMI_VIS', 'max NMI_VIS', 'mean NMI_VIS', 'min PSNR_IR', 'max PSNR_IR', 'mean PSNR_IR', 'min SSIM_IR', 'max SSIM_IR', 'mean SSIM_IR', 'min NMI_IR', 'max NMI_IR', 'mean NMI_IR']
    # # for i in range(6):
    # #     axs[1,i].boxplot(box_data[i*3:(i+1)*3])
    # #     axs[1,i].set_title(labels[i*3])
    # #     axs[1,i].set_xticklabels(['min', 'max', 'mean'])
    # #     axs[1,i].set_ylabel(labels[i*3])
    # #     if i ==0 or i == 3:
    # #         axs[1,i].set_ylim(40,100)
    # #     elif i ==1 or i == 4:
    # #         axs[1,i].set_ylim(0,1)
    # #     elif i == 2 or i == 5:
    # #         axs[1,i].set_ylim(0.9,1.8)



    


if __name__ == "__main__":
    main()
