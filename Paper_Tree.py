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
        
        self.loss_func = nn.CrossEntropyLoss()
        
        # Adjust learning rate based on depth
        self.optimizer = optim.AdamW(self.model.parameters(), weight_decay=0.01)

        
        # 学习率调整策略 OneCycleLR：
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=0.0025,
            total_steps=global_vars.num_epochs,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )

        self.left_buffer = {'x': [], 'labels': []}
        self.right_buffer = {'x': [], 'labels': []}         
         
    
    def forward(self, x, labels):
        
        outputs = self.model(x)

        # 计算标签函数的结果
        judge_tensor = torch.tensor(self.judge, device=labels.device)
        mask = torch.isin(labels, judge_tensor)
        
        # 使用标签函数计算true_labels
        true_labels = self.label_func(labels).long()
        
        # 根据输出进行预测
        _, predictions = torch.max(outputs, dim=1)
        predictions = predictions.unsqueeze(1)

        # 如果judge的长度为2，证明是叶子节点
        if len(self.judge) == 2:
            # 使用 judge 进行最终标签选择
            judge_tensor = torch.tensor(self.judge, device=outputs.device, dtype=torch.long)
            selected_labels = judge_tensor[predictions.squeeze()]
            correct = (selected_labels == labels).sum().item()
            global_vars.update_image_probabilities(x, self.judge, outputs)
        else:
            # 直接比较预测结果和标签函数的结果
            correct = (predictions == true_labels.unsqueeze(1)).sum().item()

        
        total = labels.size(0)
        global_vars.update_stats(self.chinese_name, correct, total)
        
        if self.training:
            # Only calculate loss for labels in judge_set
            filtered_outputs = outputs[mask]
            filtered_true_labels = true_labels[mask]
            if filtered_outputs.size(0) > 0:
                loss = self.loss_func(filtered_outputs, filtered_true_labels)
                # self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # self.optimizer.step()

        left_mask = (predictions.squeeze() == 0) | ((outputs[:, 0] > 0.01) & (outputs[:, 0] < 0.99))
        right_mask = (predictions.squeeze() == 1) | ((outputs[:, 0] > 0.01) & (outputs[:, 0] < 0.99))
    
        if self.left:
            self.left_buffer = self.process_buffer(self.left_buffer, self.left, x[left_mask], labels[left_mask], not self.training)

        if self.right:
            self.right_buffer = self.process_buffer(self.right_buffer, self.right, x[right_mask], labels[right_mask], not self.training)
        
    def process_buffer(self, buffer, child_node, x, labels, is_testing):
        # Directly process the incoming data
        if x.size(0) > 0:
            child_node(x, labels)
        
        # Return an empty buffer since we're not accumulating data anymore
        return {'x': [], 'labels': []}


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
        self.root = DecisionNode(is_industrial, "工业vs自然", judge=[0,1,2,3,4,5,6,7,8,9], depth=0,
            left=DecisionNode(is_land, "陆地vs天空", judge=[0,1,8,9], depth=1,
                left=DecisionNode(is_plane, "飞机vs船", judge=[8,0], depth=2, left=None, right=None),
                right=DecisionNode(is_car, "汽车vs卡车", judge=[9,1], depth=2, left=None, right=None)),
            right=DecisionNode(is_four_legged, "其他vs四足动物", judge=[2,3,4,5,6,7], depth=1,
                left=DecisionNode(is_bird, "鸟vs青蛙", judge=[2,6], depth=2, left=None, right=None),
                right=DecisionNode(is_catdog_deerhorse, "猫狗vs鹿马", judge=[3,4,5,7], depth=2,
                    left=DecisionNode(is_cat, "猫vs狗", judge=[3,5], depth=3, left=None, right=None),
                    right=DecisionNode(is_deer, "鹿vs马", judge=[4,7], depth=3, left=None, right=None)
                ),
            )
        )

    
    def zero_optimizers(self):
        self._zero_optimizers_recursive(self.root)

    def _zero_optimizers_recursive(self, node):
        if node is not None:
            node.optimizer.zero_grad()
            self._zero_optimizers_recursive(node.left)
            self._zero_optimizers_recursive(node.right)

    def step_optimizers(self):
        self._step_optimizers_recursive(self.root)

    def _step_optimizers_recursive(self, node):
        if node is not None:
            node.optimizer.step()
            self._step_optimizers_recursive(node.left)
            self._step_optimizers_recursive(node.right)

    def step_schedulers(self):
        self._step_schedulers_recursive(self.root)

    def _step_schedulers_recursive(self, node):
        if node is not None:
            node.scheduler.step()
            self._step_schedulers_recursive(node.left)
            self._step_schedulers_recursive(node.right)

    def forward(self, x, label):
        return self.root(x, label)