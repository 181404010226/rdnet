import torch
from Paper_Tree import DecisionTree
from Paper_global_vars import global_vars
from Paper_DataSet import valid_data,loader_train
from torch import optim
import torch.nn.functional as F
import os
import numpy as np
import random
import csv
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
root = os.path.join(os.path.dirname(__file__), "CIFAR10RawData")

# 初始化模型并移至GPU
model = DecisionTree().to(device)

optimizer = optim.AdamW(model.parameters(), weight_decay=0.001)

scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=0.005,
            total_steps=global_vars.num_epochs,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
        )

best_models = []
best_accuracies = []

# 初始化 GradScaler
scaler = GradScaler()

for epoch in range(global_vars.num_epochs):
    # 训练阶段
    model.train()
    train_correct = 0
    train_total = 0
    for batch_idx, (data, target) in enumerate(loader_train):
        global_vars.initialize_image_probabilities(10)
        data, target = data.to(device), target.to(device)
        
        # 使用 autocast 上下文管理器
        with autocast():
            model(data)
            
            # 计算批次损失
            predicted_probs = global_vars.image_probabilities[:len(data)]
        
            # 使用 KL 散度作为损失函数
            batch_loss = F.kl_div(predicted_probs.log(), target, reduction='batchmean')
            
            # 统计训练正确率（如果需要）
            # predicted_labels = predicted_probs.argmax(dim=1)
            # train_correct += (predicted_labels == target).sum().item()
            # train_total += len(target)

        # 使用 scaler 来缩放损失并执行反向传播
        scaler.scale(batch_loss).backward()
        # 添加梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # 打印每10个batch的loss和键值对统计
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(loader_train)}: Loss: {batch_loss.item():.4f}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    scheduler.step()

    # 输出训练阶段正确率
    # train_accuracy = train_correct / train_total if train_total > 0 else 0
    # print(f"Epoch {epoch+1}/{global_vars.num_epochs} - Train Accuracy: {train_accuracy:.4f}")

    # 测试代码
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_data):
            global_vars.initialize_image_probabilities(10)
            data, target = data.to(device), target.to(device)
            model(data)
            
            # 获取当前批次的预测概率
            predicted_probs = global_vars.image_probabilities[:len(data)]
            
            # 获取预测标签
            predicted_labels = predicted_probs.argmax(dim=1)
            
            # 计算正确预测的数量
            total_correct += (predicted_labels == target).sum().item()
            total_samples += len(target)

        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")

        # 保存前十个最佳模型
        if len(best_models) < 10 or accuracy > min(best_accuracies):
            model_state = model.state_dict()
            if len(best_models) == 10:
                # 移除准确率最低的模型
                min_acc_index = best_accuracies.index(min(best_accuracies))
                min_acc = best_accuracies[min_acc_index]
                
                # 删除文件系统中的模型文件
                for filename in os.listdir("best_models"):
                    if filename.endswith(f"acc_{min_acc:.4f}.pth"):
                        os.remove(os.path.join("best_models", filename))
                        print(f"Removed file: {filename}")
                
                best_models.pop(min_acc_index)
                best_accuracies.pop(min_acc_index)
            
            best_models.append(model_state)
            best_accuracies.append(accuracy)
            
            # 按准确率降序排序
            best_models, best_accuracies = zip(*sorted(zip(best_models, best_accuracies), 
                                                    key=lambda x: x[1], reverse=True))
            best_models = list(best_models)
            best_accuracies = list(best_accuracies)
            
            # 保存模型和优化器
            save_path_model = os.path.join("best_models", f"model_epoch_{epoch+1}_acc_{accuracy:.4f}.pth")
            save_path_optimizer = os.path.join("best_models", f"optimizer_epoch_{epoch+1}_acc_{accuracy:.4f}.pth")
            os.makedirs("best_models", exist_ok=True)
            torch.save(model_state, save_path_model)
            torch.save(optimizer.state_dict(), save_path_optimizer)
            print(f"Saved model to {save_path_model}")
            print(f"Saved optimizer to {save_path_optimizer}")


    # 训练结束后，打印最佳模型信息
    print("\nTop 10 Best Models:")
    for i, (_, acc) in enumerate(zip(best_models, best_accuracies), 1):
        print(f"{i}. Accuracy: {acc:.4f}")

    
