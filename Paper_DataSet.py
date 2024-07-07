import os
from timm.data import create_loader
from torchvision import datasets, transforms
from Paper_global_vars import global_vars

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# 定义数据配置
data_config = {
    'input_size': (3, 32, 32),
    'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN,
    'std': IMAGENET_DEFAULT_STD,
    'crop_pct': 0.96,
}

# 选择数据集
root = os.path.join(os.path.dirname(__file__), "CIFAR10RawData")
testset = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

# 使用 create_loader 创建数据加载器
valid_data = create_loader(
    testset,
    input_size=data_config['input_size'],
    batch_size=global_vars.batch_size,
    is_training=False,
    use_prefetcher=False,  # 根据需要调整
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=4,  # 根据需要调整
    distributed=False,  # 根据需要调整
    crop_pct=data_config['crop_pct'],
    pin_memory=True,  # 根据需要调整
)