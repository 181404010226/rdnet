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
    def __init__(self, model_path,dim,depth,kernel_size,patch_size):
        super(BinaryConvMixer, self).__init__()
        self.model = ConvMixer(dim,depth,kernel_size,patch_size).to(device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            # Remove the 'module.' prefix from state dict keys
            if model_path.endswith('.tar'):
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()}
            else:
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# Replace all specific network classes with BinaryConvMixer
IndustrialVsNaturalNet = lambda: BinaryConvMixer("",128,4,5,2)
LandVsSkyNet = lambda: BinaryConvMixer("",128,4,5,2)
PlaneVsShipNet = lambda: BinaryConvMixer("",128,4,5,2)
CarVsTruckNet = lambda: BinaryConvMixer("",128,4,5,2)
FourLeggedVsOthersNet = lambda: BinaryConvMixer("",128,4,5,2)
CatDogVsDeerHorseNet = lambda: BinaryConvMixer("",128,4,5,2)
CatVsDogNet = lambda: BinaryConvMixer("",128,4,5,2)
DeerVsHorseNet = lambda: BinaryConvMixer("",128,4,5,2)
BirdVsFrogNet = lambda: BinaryConvMixer("",128,4,5,2)

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