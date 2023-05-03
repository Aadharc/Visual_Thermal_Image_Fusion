import torch
from torchvision.utils import save_image
import torchvision.ops as ops
import numpy as np

def save_generated_image(y_fake, folder):
    save_image(y_fake, folder + f"yolo.png")


def save_some_examples(gen, val_loader, epoch, folder, device):
    for batch in val_loader:
        x = batch['image_vis']
        y = batch['image_ir']
        a = batch['target_vis']
        b = batch['target_ir']
        # x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        x, y = x.to(device), y.to(device)
        gen.eval()
        with torch.no_grad():
            y_fake, x_a, y_a = gen(x, y)
            save_image(y_fake, folder + f"/Fused_{epoch}.png")
            save_image(x, folder + f"/VIS_{epoch}.png")
            save_image(y, folder + f"/IR_{epoch}.png")
            save_image(x_a, folder + f"/Vis_attn_{epoch}.png")
            save_image(y_a, folder + f"/IR_attn_{epoch}.png")
        gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
