import torch
import torch.nn as nn
from GenAttn import CrossAttention, AttentionDownSampled

class CNNBlock(nn.Module):
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


class Discriminator_attn(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 128, 256, 512]):
        super().__init__()
        self.attn = AttentionDownSampled(3,32)
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
        x_a, y_a, l_a = self.attn(x, y)
        x_a = x * x_a
        y_a = y * y_a
        if y.shape[1] == 1:
            a = torch.cat([x_a, y_a], dim=1)
            a = self.initial1(a)
        else:
            a = torch.cat([x_a, y_a], dim=1)
            a = self.initial(a)
        a = self.model(a)
        return a


def test():
    x = torch.randn((1, 3, 512, 512))
    y = torch.randn((1, 3, 512, 512))
    model = Discriminator_attn(in_channels=3)
    preds = model(x, y)
    print(preds)
    print(preds.shape)


if __name__ == "__main__":
    test()