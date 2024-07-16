import torch
import torch.nn as nn
import torch.optim as optim
from Paper_global_vars import global_vars
from Paper_Network import get_network
import numpy as np

class DecisionNode(nn.Module):
    def __init__(self, chinese_name, judge=[-1,-1], left=None, right=None, depth=0, base_lr=0.001):
        super(DecisionNode, self).__init__()
        self.model = get_network(chinese_name)
        self.left = left
        self.right = right
        self.chinese_name = chinese_name
        self.judge = judge      
    
    def forward(self, x,labels=None):
        outputs = self.model(x)

        global_vars.update_image_probabilities(self.judge, outputs)

        if self.left:
            self.left(x)
        if self.right:
            self.right(x)




class DecisionTree(nn.Module):
    def __init__(self):
        super(DecisionTree, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # 构建决策树
        self.root = DecisionNode("工业vs自然", judge=[[0,1,8,9],[2,3,4,5,6,7]], depth=0,
            left=DecisionNode("陆地vs天空", judge=[[0,8],[1,9]], depth=1,
                left=DecisionNode("飞机vs船", judge=[[0],[8]], depth=2, left=None, right=None),
                right=DecisionNode("汽车vs卡车", judge=[[1],[9]], depth=2, left=None, right=None)),
            right=DecisionNode("其他vs四足动物", judge=[[2,6],[3,4,5,7]], depth=1,
                left=DecisionNode("鸟vs青蛙", judge=[[2],[6]], depth=2, left=None, right=None),
                right=DecisionNode("猫狗vs鹿马", judge=[[3,5],[4,7]], depth=2,
                    left=DecisionNode("猫vs狗", judge=[[3],[5]], depth=3, left=None, right=None),
                    right=DecisionNode("鹿vs马", judge=[[4],[7]], depth=3, left=None, right=None)
                ),
            )
        )

    def forward(self, x):

        return self.root(x)