import torch
from Paper_TreeForTest import SequentialDecisionTree
from Paper_global_vars import global_vars
from Paper_DataSet import valid_data
import os
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

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

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

clear_directory('/root/autodl-tmp')


# 初始化模型并移至GPU
model = SequentialDecisionTree().to(device)

# 加载模型
model_path = 'best_models/model_epoch_444_acc_0.9630.pth'
model.load_state_dict(torch.load(model_path, map_location=device))


# 测试代码
model.eval()
total_correct = 0

confusion_dict = defaultdict(int)
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


with torch.no_grad():
    for batch_idx, (data, target) in enumerate(valid_data):
        data, target = data.to(device), target.to(device)
        model(data)
        
        # 遍历当前batch中的每个样本
        for idx, true_label in enumerate(target):
            predicted_probs = global_vars.log_image_probabilities[idx]
            predicted_label = predicted_probs.argmax().item()
            is_correct = predicted_label == true_label.item()
            total_correct += is_correct

            if not is_correct:
                confusion_pair = (class_labels[true_label.item()], class_labels[predicted_label])
                confusion_dict[confusion_pair] += 1

                
            if not is_correct or batch_idx == 0:
                # Create two separate figures
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                fig2, ax3 = plt.subplots(figsize=(10, 5))
                
                fig1.suptitle(f'Batch {batch_idx}, Sample {idx}, True: {class_labels[true_label.item()]}, Pred: {class_labels[predicted_label]}')
                
                # 显示原始图片
                img = data[idx].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]
                ax1.imshow(img)
                ax1.set_title('Raw Image')
                ax1.axis('off')
                
                # 绘制概率分布图
                probs_np = predicted_probs.cpu().numpy()
                bars = ax2.bar(class_labels, probs_np)
                ax2.set_title('Probability Distribution')
                ax2.set_xlabel('Class')
                ax2.set_ylabel('Probability')
                ax2.tick_params(axis='x', rotation=45)
                
                # 在柱状图上方标记真实概率
                for bar, prob in zip(bars, probs_np):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{prob:.4f}', ha='center', va='bottom')
                
                # Node probabilities plot
                ax3.set_title('Node Probabilities')
                ax3.set_xlabel('Node')
                ax3.set_ylabel('Probability')
                
                node_names = []
                left_probs = []
                right_probs = []
                
                def traverse_tree(nodes):
                    for node in nodes:
                        if node and idx in node.node_probabilities:
                            node_names.append(node.english_name)
                        left_prob, right_prob = node.node_probabilities[idx][0]
                        left_probs.append(left_prob)
                        right_probs.append(right_prob)
                
                traverse_tree(model.nodes)
                
                x = range(len(node_names))
                width = 0.35
                left_bars = ax3.bar([i - width/2 for i in x], left_probs, width, label='Left')
                right_bars = ax3.bar([i + width/2 for i in x], right_probs, width, label='Right')
                ax3.set_xticks(x)
                ax3.set_xticklabels(node_names, rotation=45, ha='right')
                ax3.legend()
                
                # Add accuracy labels to the bars
                def add_accuracy_labels(bars):
                    for bar in bars:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width() / 2, height,
                                f'{height:.2f}', ha='center', va='bottom')
                
                add_accuracy_labels(left_bars)
                add_accuracy_labels(right_bars)
                
                plt.tight_layout()
                
                # Save the two figures separately
                result = "correct" if is_correct else "error"
                fig1.savefig(f'/root/autodl-tmp/{result}_analysis_batch_{batch_idx}_sample_{idx}_1.png')
                fig2.savefig(f'/root/autodl-tmp/{result}_analysis_batch_{batch_idx}_sample_{idx}_2.png')
                plt.close(fig1)
                plt.close(fig2)
            

    accuracy = total_correct /  len(valid_data.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")



# Create and save pie chart
total_confusions = sum(confusion_dict.values())
confusion_percentages = {k: v / total_confusions * 100 for k, v in confusion_dict.items()}

# Sort confusions by percentage and combine small categories
sorted_confusions = sorted(confusion_percentages.items(), key=lambda x: x[1], reverse=True)
pie_data = []
pie_labels = []
other_percentage = 0
threshold = 1

for (true_label, pred_label), percentage in sorted_confusions:
    if percentage >= threshold:
        pie_data.append(percentage)
        pie_labels.append(f"{true_label} → {pred_label}")
    else:
        other_percentage += percentage

if other_percentage > 0:
    pie_data.append(other_percentage)
    pie_labels.append("Others")

# Create a colors list
base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = (base_colors * ((len(pie_data) - 1) // len(base_colors) + 1))[:len(pie_data) - 1]
if other_percentage > 0:
    colors.append('#999999')  # Gray color for "Others"

# ... existing code ...

plt.figure(figsize=(12, 8))
wedges, texts, autotexts = plt.pie(pie_data, labels=None, autopct='%1.1f%%', startangle=90, 
                                   wedgeprops=dict(width=0.6), textprops=dict(color="k"),
                                   colors=colors)

# Add lines connecting wedges to labels
for i, wedge in enumerate(wedges):
    ang = (wedge.theta2 + wedge.theta1) / 2
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    
    plt.annotate(pie_labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                 horizontalalignment=horizontalalignment,
                 verticalalignment="center",
                 arrowprops=dict(arrowstyle="-", color="0.5",
                                 connectionstyle=connectionstyle))

plt.title(f"Confusion Distribution (>{threshold}%)")
plt.axis('equal')
plt.savefig(f'/root/autodl-tmp/confusion_pie_chart_{threshold}.png', bbox_inches='tight')
plt.close()