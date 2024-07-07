import torch
from Paper_Tree import DecisionTree
from Paper_global_vars import global_vars
from Paper_DataSet import valid_data,loader_train
from torch import optim
import torch.nn.functional as F
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
optimizer = model.global_optimizer
scheduler = model.global_scheduler

for epoch in range(global_vars.num_epochs):
    # 训练阶段
    model.train()
    for batch_idx, (data, target) in enumerate(loader_train):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.zero_optimizers()  
        model(data, target.squeeze())
        
        batch_loss = 0
        valid_samples = 0
        
        for img, true_label in zip(data, target):
            img_key = tuple(img.cpu().flatten().tolist())
            if img_key in global_vars.image_probabilities:
                probs = global_vars.image_probabilities[img_key]
                valid_samples += 1
                predicted_probs = torch.zeros(10, device=device)
                for i in range(10):
                    predicted_probs[i] = probs.get(i, torch.tensor(0.0, device=device)).requires_grad_()
                sample_loss = F.cross_entropy(predicted_probs.unsqueeze(0), true_label.unsqueeze(0))
                batch_loss += sample_loss
        
        if valid_samples >= 1:  # 确保有足够的有效样本
            batch_loss /= valid_samples
            batch_loss.backward()
            optimizer.step() 
            
            # 统计键值对数量
            pair_counts = {}
            for probs in global_vars.image_probabilities.values():
                num_pairs = len(probs)
                pair_counts[num_pairs] = pair_counts.get(num_pairs, 0) + 1
            
            # 打印每个batch的loss和键值对统计
            print(f"Batch {batch_idx+1}/{len(loader_train)}: Loss: {batch_loss.item():.4f}, Valid samples: {valid_samples}")
            print("Key-value pair counts:")
            for num_pairs in sorted(pair_counts.keys(), reverse=True):
                print(f"  {num_pairs} pairs: {pair_counts[num_pairs]} images")
        
        model.step_optimizers() 
        global_vars.image_probabilities.clear()
     
    scheduler.step()
    model.step_schedulers()  
    
    # 每个epoch结束后打印统计信息和学习率
    print(f"Epoch {epoch+1}/{global_vars.num_epochs}:, Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    global_vars.print_all_stats()
    global_vars.reset_stats()

    # 测试代码
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

    global_vars.image_probabilities.clear()
    global_vars.reset_stats()
