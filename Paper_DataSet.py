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
trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
testset = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

# 创建训练数据加载器
loader_train = create_loader(
    trainset,
    input_size=data_config['input_size'],
    batch_size=global_vars.train_batch_size,  # -b 256
    is_training=True,
    use_prefetcher=False,
    no_aug=False,
    re_prob=0.25,  # --reprob 0.25
    re_mode='pixel',  # --remode pixel
    re_count=1,
    scale=(0.75, 1.0),  # --scale 0.75 1.0
    ratio=(3./4., 4./3.),
    hflip=0.5,
    vflip=0.,
    color_jitter=0.4,
    auto_augment='rand-m9-mstd0.5-inc1',  # --aa rand-m9-mstd0.5-inc1
    num_aug_splits=0,
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=8,  # -j 8
    distributed=False,
    crop_pct=data_config['crop_pct'],
    collate_fn=None,
    use_multi_epochs_loader=False,
    worker_seeding='all',
    pin_memory=True,
)
# 使用 create_loader 创建数据加载器
valid_data = create_loader(
    testset,
    input_size=data_config['input_size'],
    batch_size=global_vars.test_batch_size,
    is_training=False,
    use_prefetcher=False,  # 根据需要调整
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=8,  # 根据需要调整
    distributed=False,  # 根据需要调整
    crop_pct=data_config['crop_pct'],
    pin_memory=True,  # 根据需要调整
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import torch

    # 获取一批训练数据
    data_iter = iter(loader_train)
    images, labels = next(data_iter)

    # 存储64张训练图片
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images[:64]):
        img = img.permute(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
        img = img.numpy()
        
        # Normalize the image data to 0-1 range
        img = (img - img.min()) / (img.max() - img.min())
        
        plt.imsave(os.path.join(save_dir, f"image_{i}.png"), img)