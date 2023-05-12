import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VisionTransformer


class block(nn.Module):
    def __init__(self, in_chan , out_chan, down = True, act = "prelu", use_dropout = False):
        super(block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias = False, padding_mode= 'reflect')
            if down
            else nn.ConvTranspose2d(in_chan, out_chan, 4, 2, 1, bias = False),
            nn.InstanceNorm2d(out_chan, affine = True),
            nn.PReLU() if act == "prelu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
class convblock(nn.Module):
    def __init__(self, in_chan = 3, features = 32):
        super().__init__()
        self.down1 = block(in_chan, features, down = True, act = "prelu", use_dropout= False)
        self.down2 = block(features, features, down = True, act = "prelu", use_dropout= False)
        self.down3 = block(features, features, down = True, act = "prelu", use_dropout= False)
        self.down4 = block(features, features, down = True, act = "prelu", use_dropout= False)
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x
    


class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.in_channels = in_channels

        # self.modality_id_ir = modality_id_ir
        # self.modality_id_vis = modality_id_vis

        # Convolutional layers for extracting query, key, and value features
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Convolutional layer for combining the attended features
        self.combine_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

    def forward(self, x1, x2):
        # Compute the query, key, and value features from x1 and x2
        batch_size, _, h, w = x1.size()
        query1 = self.query_conv(x1).view(batch_size, -1, h*w).permute(0, 2, 1)
        # print(f"query1 shape {query1.shape}")
        # query1 = query1 + self.modality_id_vis
        key1 = self.key_conv(x1).view(batch_size, -1, h*w)
        value1 = self.value_conv(x1).view(batch_size, -1, h*w)
        query2 = self.query_conv(x2).view(batch_size, -1, h*w).permute(0, 2, 1)
        # query2 = query2 + self.modality_id_ir
        key2 = self.key_conv(x2).view(batch_size, -1, h*w)
        value2 = self.value_conv(x2).view(batch_size, -1, h*w)

        # Compute the attention map and attended features
        attn1 = torch.bmm(query1, key1)
        attn1 = F.softmax(attn1, dim=2)
        attn1 = torch.clamp(attn1, 0, 1)
        attended1 = torch.bmm(value2, attn1.permute(0, 2, 1)).view(batch_size, self.in_channels, h, w)

        # Combine the attended features from x1 and x2
        combined1 = self.combine_conv(torch.cat((x1, attended1), dim=1))

        # Compute the attention map and attended features
        attn2 = torch.bmm(query2, key2)
        attn2 = F.softmax(attn2, dim=2)
        attn2 = torch.clamp(attn2, 0, 1)
        attended2 = torch.bmm(value1, attn2.permute(0, 2, 1)).view(batch_size, self.in_channels, h, w)

        # Combine the attended features from x1 and x2
        combined2 = self.combine_conv(torch.cat((x1, attended2), dim=1))

        # return torch.cat((torch.abs(combined1).sum(dim=1).unsqueeze(1), torch.abs(combined2).sum(dim = 1).unsqueeze(1)), dim = 1)
        return (combined1.pow(2).mean(1).unsqueeze(1), combined2.pow(2).mean(1).unsqueeze(1))

class convup(nn.Module):
    def __init__(self, in_chan = 3, features = 32):
        super().__init__()
        # self.enc = convblock(in_chan= in_chan, features = 32)
        self.up1 = block( 2 + 32 + 32, features * 2, down = False, act = "prelu", use_dropout= False)
        self.up2 = block(features * 2, features * 4, down = False, act = "prelu", use_dropout= False)
        self.up3 = block(features * 4, features, down = False, act = "prelu", use_dropout= False)
        self.up4 = block(features , in_chan, down = False, act = "prelu", use_dropout= False)
    def forward(self, attn1, attn2,x, y):
        # d1 = self.enc.down1(x)
        # d2 = self.enc.down2(d1)
        # d3 = self.enc.down3(d2)
        # d4 = self.enc.down4(d3)
        u1 = self.up1(torch.cat((attn1, attn2, x, y), 1))
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u4 = self.up4(u3)
        return u4


class AttentionDownSampled(nn.Module):
    def __init__(self, in_chan = 3, features = 32):
        super().__init__()
        self.down1 = block(in_chan, features, down = True, act = "prelu", use_dropout= False)
        self.down2 = block(features, features, down = True, act = "prelu", use_dropout= False)
        self.down3 = block(features, features, down = True, act = "prelu", use_dropout= False)
        self.down4 = block(features, features, down = True, act = "prelu", use_dropout= False)
        # self.attn = CrossAttention(32,  modality_id_vis=torch.randn(8, 2048, 4).to(config.DEVICE),  modality_id_ir=torch.randn(8, 2048, 4).to(config.DEVICE))
        self.attn = CrossAttention(32)
    def forward(self, x, y):
        # print(x.shape)
        h,w = x.shape[2], x.shape[3]
        dx1 = self.down1(x)
        dx2 = self.down2(dx1)
        dx3 = self.down3(dx2)
        dx4 = self.down4(dx3)

        dy1 = self.down1(y)
        dy2 = self.down2(dy1)
        dy3 = self.down3(dy2)
        dy4 = self.down4(dy3)

        attn1, attn2 = self.attn(dx4, dy4)

        attn1 = torch.nn.functional.interpolate(attn1, (h,w), mode = 'bilinear', align_corners = False)
        attn2 = torch.nn.functional.interpolate(attn2, (h,w), mode = 'bilinear', align_corners = False)

        # attn1 = torch.clamp(attn1, 0, 1)
        # attn2 = torch.clamp(attn2, 0, 1)

        return x * attn1, y * attn2
        # return attn1, attn2


class Generator_attn(nn.Module):
    def __init__(self, in_chan = 3, features = 32):
        super().__init__()
        self.attn = AttentionDownSampled(3,32)
        self.down1 = block(in_chan, features, down = True, act = "prelu", use_dropout= False)
        self.down2 = block(features, features, down = True, act = "prelu", use_dropout= False)
        self.down3 = block(features, features, down = True, act = "prelu", use_dropout= False)
        self.down4 = block(features, features, down = True, act = "prelu", use_dropout= False)
        # self.attn = CrossAttention(32)
        self.up1 = block( 32 + 32, features * 2, down = False, act = "prelu", use_dropout= False)

        # with noise
        self.up2 = block(features * 2 + 32, features * 4, down = False, act = "prelu", use_dropout= False)
        self.up3 = block(features * 4 + 32, features, down = False, act = "prelu", use_dropout= False)
        
        # without noise
        # self.up2 = block(features * 2, features * 4, down = False, act = "prelu", use_dropout= False)
        # self.up3 = block(features * 4, features, down = False, act = "prelu", use_dropout= False)
        self.up4 = block(features , in_chan, down = False, act = "prelu", use_dropout= False)

    def forward(self, x, y):
        # w,h = x.shape[2], x.shape[3]
        x_a, y_a = self.attn(x,y)
        # print(f"x_a shape {x_a.shape}")
        dx1 = self.down1(x_a)
        dx2 = self.down2(dx1)
        dx3 = self.down3(dx2)
        dx4 = self.down4(dx3)

        dy1 = self.down1(y_a)
        dy2 = self.down2(dy1)
        dy3 = self.down3(dy2)
        dy4 = self.down4(dy3)

        # attn1, attn2 = self.attn(dx4, dy4)

        # attn1 = torch.nn.functional.interpolate(attn1, (h,w), mode = 'bilinear', align_corners = False)
        # attn2 = torch.nn.functional.interpolate(attn2, (h,w), mode = 'bilinear', align_corners = False)

        # print(f"attn shape {attn1.shape}, encoded shape {dy4.shape}")

        # u1 = self.up1(torch.cat((attn1, attn2, dx4, dy4), 1))    # used before scaling the attention up

        u1 = self.up1(torch.cat((dx4, dy4), 1))

        # min noise of both modality
        # u2 = self.up2(torch.cat((u1, torch.mean(torch.stack((dx3, dy3)), dim=0)[0]),1))
        # u3 = self.up3(torch.cat((u2, torch.mean(torch.stack((dx2, dy2)), dim=0)[0]),1))

        # only thermal noise
        u2 = self.up2(torch.cat((u1, dy3),1))
        u3 = self.up3(torch.cat((u2, dy2),1))

        # no noise
        # u2 = self.up2(u1)
        # u3 = self.up3(u2)
        u4 = self.up4(u3) 
        return u4, x_a, y_a




class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine= True),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Gen(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.attn = AttentionDownSampled(3,32)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels*2, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=False)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, y):
        x_a, y_a = self.attn(x, y)

        d = torch.cat([x_a,y_a], dim=1)
        # print("d shape", d.shape)
        d1 = self.initial_down(d)
        # print("d1 shape", d1.shape)
        d2 = self.down1(d1)
        # print("d2 shape", d2.shape)
        d3 = self.down2(d2)
        # print("d3 shape", d3.shape)
        d4 = self.down3(d3)
        # print("d4 shape", d4.shape)
        d5 = self.down4(d4)
        # print("d5 shape", d5.shape)
        d6 = self.down5(d5)
        # print("d6 shape", d6.shape)
        d7 = self.down6(d6)
        # print("d7 shape", d7.shape)
        bottleneck = self.bottleneck(d7)
        # print("bottle shape", bottleneck.shape)
        up1 = self.up1(bottleneck)
        # print("up1 shape", up1.shape)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1)), x_a, y_a



def test():
    x = torch.randn((8, 3, 512, 640))
    y = torch.randn((8, 3, 512, 640))
    # model = CrossAttention(in_channels=32)
    model = Generator_attn(3,32)
    preds = model(x,y)
    print(preds.shape)


if __name__ == "__main__":
    test()