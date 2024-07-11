import torch
from Paper_Tree import DecisionTree
from Paper_global_vars import global_vars
from Paper_DataSet import valid_data
import os
import numpy as np
import random
import matplotlib.pyplot as plt

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

# 加载模型
model_path = 'best_models/model_epoch_134_acc_0.9277.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

# 测试代码
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(valid_data):
        global_vars.image_probabilities.clear()
        data, target = data.to(device), target.to(device)
        model(data)
        
        # 遍历当前batch中的每个样本
        for idx, true_label in enumerate(target):
            if idx in global_vars.image_probabilities:
                probs = global_vars.image_probabilities[idx]
                predicted_probs = torch.zeros(10, device=device)
                for i in range(10):
                    predicted_probs[i] = probs.get(i, 0.0)
                total_correct += predicted_probs[true_label.item()].item()
                
                # 创建一个包含两个子图的图形
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                fig.suptitle(f'Batch {batch_idx}, Sample {idx}, True Label: {true_label.item()}')
                
                # 显示原始图片
                img = data[idx].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]
                ax1.imshow(img)
                ax1.set_title('Original Image')
                ax1.axis('off')
                
                # 绘制概率分布图
                probs_np = predicted_probs.cpu().numpy()
                bars = ax2.bar(range(10), probs_np)
                ax2.set_title('Probability Distribution')
                ax2.set_xlabel('Class')
                ax2.set_ylabel('Probability')
                
                # 在柱状图上方标记真实概率
                for bar, prob in zip(bars, probs_np):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{prob:.4f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f'/root/autodl-tmp/prob_dist_batch_{batch_idx}_sample_{idx}.png')
                plt.close()
                
            total_samples += 1

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")