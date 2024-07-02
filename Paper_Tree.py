import torch
import torch.nn as nn
import torch.optim as optim
from Paper_global_vars import global_vars
from Paper_Network import get_network
import numpy as np

class DecisionNode(nn.Module):
    def __init__(self, label_func, chinese_name, judge=[-1,-1], left=None, right=None, depth=0, base_lr=0.001):
        super(DecisionNode, self).__init__()
        self.model = get_network(chinese_name)
        self.label_func = label_func
        self.left = left
        self.right = right
        self.chinese_name = chinese_name
        self.judge = judge  
        self.loss_func = nn.BCELoss()
        
        # Adjust learning rate based on depth
        self.learning_rate = base_lr / (2 ** depth)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 学习率调整策略 MultiStep：
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                milestones=[int(global_vars.num_epochs * 0.56), int(global_vars.num_epochs * 0.78)],
                                                gamma=0.1, last_epoch=-1)
        self.left_buffer = {'x': [], 'labels': []}
        self.right_buffer = {'x': [], 'labels': []}         
         
    
    def forward(self, x, labels):

        outputs = self.model(x)

        # debug：只跑一个
        # if self.chinese_name != '工业vs自然':
        #     return 
        # if 1 == 1:
        #     import matplotlib.pyplot as plt
        #     import os
        #     if not os.path.exists('images'):
        #         os.makedirs('images')
        #     for i in range(x.size(0)):
        #         img = x[i].cpu().numpy().transpose((1, 2, 0))
        #         img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        #         img = np.clip(img, 0, 1)
        #         plt.imsave(f'images/{self.chinese_name}img_{i}.png', img)
        
        # 计算标签函数的结果
        true_labels = self.label_func(labels).float().unsqueeze(1)
        
        # 根据输出进行预测
        _, predictions = torch.max(outputs, dim=1)
        predictions = predictions.unsqueeze(1)
 
        if self.judge != [-1, -1]:
            # 使用 judge 进行最终标签选择
            judge_tensor = torch.tensor(self.judge, device=outputs.device, dtype=torch.long)
            selected_labels = judge_tensor[predictions.squeeze()]
            correct = (selected_labels == labels).sum().item()
        else:
            # 直接比较预测结果和标签函数的结果
            correct = (predictions == true_labels).sum().item()
        
        total = labels.size(0)
        global_vars.update_stats(self.chinese_name, correct, total)
        
        if self.training:
            loss = self.loss_func(outputs, true_labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        # 不管判断是否正确，都往正确的分支传递
        left_mask = self.label_func(labels) == 0
        right_mask = ~left_mask

        # 保证每次传递的批次量大于128
        if self.left:
            self.left_buffer['x'].append(x[left_mask])
            self.left_buffer['labels'].append(labels[left_mask])
            if sum(b.size(0) for b in self.left_buffer['x']) >= 1:
                left_x = torch.cat(self.left_buffer['x'])
                left_labels = torch.cat(self.left_buffer['labels'])
                self.left(left_x, left_labels)
                self.left_buffer = {'x': [], 'labels': []}

        if self.right:
            self.right_buffer['x'].append(x[right_mask])
            self.right_buffer['labels'].append(labels[right_mask])
            if sum(b.size(0) for b in self.right_buffer['x']) >= 1:
                right_x = torch.cat(self.right_buffer['x'])
                right_labels = torch.cat(self.right_buffer['labels'])
                self.right(right_x, right_labels)
                self.right_buffer = {'x': [], 'labels': []}


class DecisionTree(nn.Module):
    def __init__(self):
        super(DecisionTree, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 定义标签函数
        is_industrial = lambda labels: torch.where(labels.unsqueeze(1) == torch.tensor([2,3,4,5,6,7], device=device), 1, 0).sum(dim=1).bool().int()
        is_land = lambda labels: torch.where(labels.unsqueeze(1) == torch.tensor([1,9], device=device), 1, 0).sum(dim=1).bool().int()
        is_plane = lambda labels: torch.where(labels == 0, 1, 0)
        is_car = lambda labels: torch.where(labels == 1, 1, 0)
        is_four_legged = lambda labels: torch.where(labels.unsqueeze(1) == torch.tensor([3,4,5,7], device=device), 1, 0).sum(dim=1).bool().int()
        is_catdog_deerhorse = lambda labels: torch.where(labels.unsqueeze(1) == torch.tensor([4,7], device=device), 1, 0).sum(dim=1).bool().int()
        is_cat = lambda labels: torch.where(labels == 5, 1, 0)
        is_deer = lambda labels: torch.where(labels == 7, 1, 0)
        is_bird = lambda labels: torch.where(labels == 6, 1, 0)
        
        # 构建决策树
        self.root = DecisionNode(is_industrial, "工业vs自然", judge=[-1,-1], depth=0,
            left=DecisionNode(is_land, "陆地vs天空", judge=[-1,-1], depth=1,
                left=DecisionNode(is_plane, "飞机vs船", judge=[0,8], depth=2, left=None, right=None),
                right=DecisionNode(is_car, "汽车vs卡车", judge=[1,9], depth=2, left=None, right=None)),
            right=DecisionNode(is_four_legged, "其他vs四足动物", judge=[-1,-1], depth=1,
                left=DecisionNode(is_bird, "鸟vs青蛙", judge=[2,6], depth=2, left=None, right=None),
                right=DecisionNode(is_catdog_deerhorse, "猫狗vs鹿马", judge=[-1,-1], depth=2,
                    left=DecisionNode(is_cat, "猫vs狗", judge=[3,5], depth=3, left=None, right=None),
                    right=DecisionNode(is_deer, "鹿vs马", judge=[4,7], depth=3, left=None, right=None)
                ),
            )
        )

    def forward(self, x, label):
        return self.root(x, label)