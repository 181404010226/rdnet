import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=2, n_classes=2):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes),
        nn.Sigmoid()
    )

class BinaryConvMixer(nn.Module):
    def __init__(self, model_path):
        super(BinaryConvMixer, self).__init__()
        self.model = ConvMixer().to(device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            # Remove the 'module.' prefix from state dict keys
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# Replace all specific network classes with BinaryConvMixer
IndustrialVsNaturalNet = lambda: BinaryConvMixer("data/train工业vx自然/model_0.9881_epoch82.pth")
LandVsSkyNet = lambda: BinaryConvMixer("data/train飞机轮船vs汽车卡车/model_0.9888_epoch110.pth")
PlaneVsShipNet = lambda: BinaryConvMixer("data/train飞机vs轮船/model_0.9815_epoch112.pth")
CarVsTruckNet = lambda: BinaryConvMixer("data/train汽车vs卡车/model_0.9795_epoch104.pth")
FourLeggedVsOthersNet = lambda: BinaryConvMixer("data/train鸟青蛙vs四脚兽/model_0.9720_epoch102.pth")
CatDogVsDeerHorseNet = lambda: BinaryConvMixer("data/train猫狗vs马鹿/model_0.9695_epoch98.pth")
CatVsDogNet = lambda: BinaryConvMixer("data/train猫vs狗/model_0.9285_epoch589.pth")
DeerVsHorseNet = lambda: BinaryConvMixer("data/train马vs鹿/model_0.9885_epoch102.pth")
BirdVsFrogNet = lambda: BinaryConvMixer("data/train鸟vs青蛙/model_0.9830_epoch104.pth")

def get_network(node_name):
    networks = {
        "工业vs自然": IndustrialVsNaturalNet,
        "陆地vs天空": LandVsSkyNet,
        "飞机vs船": PlaneVsShipNet,
        "汽车vs卡车": CarVsTruckNet,
        "其他vs四足动物": FourLeggedVsOthersNet,
        "猫狗vs鹿马": CatDogVsDeerHorseNet,
        "猫vs狗": CatVsDogNet,
        "鹿vs马": DeerVsHorseNet,
        "鸟vs青蛙": BirdVsFrogNet
    }
    return networks[node_name]()