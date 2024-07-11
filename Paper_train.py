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
optimizer = optim.AdamW(model.parameters(), weight_decay=0.01)

scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=0.0025,
            total_steps=global_vars.num_epochs,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )

best_models = []
best_accuracies = []

# Create a CSV file for logging
csv_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Epoch', 'Batch', 'Loss', 'Valid Samples', 'Learning Rate', 'Key-Value Pair Counts'])

# 初始化 GradScaler
scaler = GradScaler()

for epoch in range(global_vars.num_epochs):
    # 训练阶段
    model.train()
    train_correct = 0
    train_total = 0
    for batch_idx, (data, target) in enumerate(loader_train):
        global_vars.image_probabilities.clear()
        data, target = data.to(device), target.to(device)
        
        # 使用 autocast 上下文管理器
        with autocast():
            model(data)
            
            batch_loss = 0
            
            for idx, (_, true_label) in enumerate(zip(data, target)):
                if idx in global_vars.image_probabilities:
                    probs = global_vars.image_probabilities[idx]
                    predicted_probs = torch.zeros(10, device=device)
                    for i in range(10):
                        predicted_probs[i] = probs.get(i, 0.0)
                    sample_loss = F.cross_entropy(predicted_probs.unsqueeze(0), true_label.unsqueeze(0))
                    batch_loss += sample_loss
                    
                    # 统计训练正确率
                    predicted_label = predicted_probs.argmax().item()
                    train_correct += (predicted_label == true_label.item())
                    train_total += 1
            
            batch_loss /= global_vars.train_batch_size

        # 使用 scaler 来缩放损失并执行反向传播
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # 统计键值对数量
        pair_counts = {}
        for probs in global_vars.image_probabilities.values():
            num_pairs = len(probs)
            pair_counts[num_pairs] = pair_counts.get(num_pairs, 0) + 1
        
        # 打印每10个batch的loss和键值对统计
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(loader_train)}: Loss: {batch_loss.item():.4f}, Learning rate: {0.002}")
            print("Key-value pair counts (sum of probabilities > 0.1):")
            pair_counts = {}
            for probs in global_vars.image_probabilities.values():
                num_pairs = sum(1 for p in probs.values() if p > 0.1)
                pair_counts[num_pairs] = pair_counts.get(num_pairs, 0) + 1
            pair_counts_str = ""
            for num_pairs in sorted(pair_counts.keys(), reverse=True):
                pair_count = f"{num_pairs} pairs: {pair_counts[num_pairs]} images"
                print(f"  {pair_count}", end=" ")
                pair_counts_str += pair_count + "; "
            print()
            # Write to CSV
            csv_writer.writerow([
                epoch + 1,
                batch_idx + 1,
                f"{batch_loss.item():.4f}",
                #f"{scheduler.get_last_lr()[0]:.6f}",
                pair_counts_str.strip()
            ])
            csv_file.flush()  # Ensure data is written immediately
    
    # scheduler.step()

    # 输出训练阶段正确率
    train_accuracy = train_correct / train_total if train_total > 0 else 0
    print(f"Epoch {epoch+1}/{global_vars.num_epochs} - Train Accuracy: {train_accuracy:.4f}")

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
                    predicted_label = predicted_probs.argmax().item()
                    total_correct += (predicted_label == true_label.item())
                total_samples += 1

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
                        print(f"Removed model file: {filename}")
                        break
                
                best_models.pop(min_acc_index)
                best_accuracies.pop(min_acc_index)
            
            best_models.append(model_state)
            best_accuracies.append(accuracy)
            
            # 按准确率降序排序
            best_models, best_accuracies = zip(*sorted(zip(best_models, best_accuracies), 
                                                    key=lambda x: x[1], reverse=True))
            best_models = list(best_models)
            best_accuracies = list(best_accuracies)
            
            # 保存模型
            save_path = os.path.join("best_models", f"model_epoch_{epoch+1}_acc_{accuracy:.4f}.pth")
            os.makedirs("best_models", exist_ok=True)
            torch.save(model_state, save_path)
            print(f"Saved model to {save_path}")


    # 训练结束后，打印最佳模型信息
    print("\nTop 10 Best Models:")
    for i, (_, acc) in enumerate(zip(best_models, best_accuracies), 1):
        print(f"{i}. Accuracy: {acc:.4f}")