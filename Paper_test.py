import torch
from Paper_Tree import DecisionTree
from Paper_global_vars import global_vars
from Paper_DataSet import valid_data
import os
import numpy as np
import random

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
root = os.path.join(os.path.dirname(__file__), "CIFAR10RawData")

# 初始化模型并移至GPU
model = DecisionTree().to(device)

# 测试阶段
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for data, target in valid_data:
        data, target = data.to(device), target.to(device)
        model(data, target)
        
        # 遍历当前batch中的每个样本
        for img, true_label in zip(data, target):
            img_key = tuple(img.cpu().flatten().tolist())
            if img_key in global_vars.image_probabilities:
                probs = global_vars.image_probabilities[img_key]
                predicted_label = max(probs, key=probs.get)
                if predicted_label == true_label.item():
                    total_correct += 1
            total_samples += 1

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")

print(f"Test:")
global_vars.print_all_stats()
global_vars.reset_stats()
