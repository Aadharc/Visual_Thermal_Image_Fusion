import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    '''
    CNN block for Discriminator

    This class defines a convolutional neural network (CNN) block designed for use in a Discriminator model. 
    The block consists of a series of operations that process input data through convolutional layers, batch 
    normalization, and a LeakyReLU activation function. It is commonly used to create the foundational layers 
    of a Discriminator network in applications like image synthesis and GANs (Generative Adversarial Networks).

    Args:
        in_channels (int): Number of input channels to the convolutional layer.
        out_channels (int): Number of output channels from the convolutional layer.
        stride (int): Stride value for the convolution operation.
    '''
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


# Discriminator for the GAN
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        self.initial1 = nn.Sequential(
            nn.Conv2d(
                4,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )
        self.initial2 = nn.Sequential(
            nn.Conv2d(
                2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        if (y.shape[1] == 1 and x.shape[1] == 3) or (x.shape[1] == 1 and y.shape[1] == 3):
            x = torch.cat([x, y], dim=1)
            x = self.initial1(x)
        elif y.shape[1] == 1 and x.shape[1] == 1:
            x = torch.cat([x,y], dim = 1)
            x = self.initial2(x)

        else:
            x = torch.cat([x, y], dim=1)
            x = self.initial(x)
        x = self.model(x)
        return x


def test():
    x = torch.randn((1, 3, 512, 512))
    y = torch.randn((1, 3, 512, 512))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()