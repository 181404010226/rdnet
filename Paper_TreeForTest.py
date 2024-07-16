import torch
import torch.nn as nn
import torch.optim as optim
from Paper_global_vars import global_vars
from Paper_Network import get_network
import numpy as np

class DecisionNode(nn.Module):
    def __init__(self, english_name, judge=[-1,-1], left=None, right=None, depth=0, base_lr=0.001):
        super(DecisionNode, self).__init__()
        self.model = get_network(english_name)
        self.left = left
        self.right = right
        self.english_name = english_name
        self.judge = judge
        self.node_probabilities = {}
    
    def forward(self, x, labels=None, sample_idx=None):
        outputs = self.model(x)

        global_vars.update_image_probabilities(self.judge, outputs)

        # Calculate probabilities for left and right
        left_prob = outputs[:, 0]
        right_prob = outputs[:, 1]
        
        # Store probabilities for each sample
        for idx, (l_prob, r_prob) in enumerate(zip(left_prob, right_prob)):
            if sample_idx is not None:
                idx = sample_idx[idx]
            if idx not in self.node_probabilities:
                self.node_probabilities[idx] = []
            self.node_probabilities[idx].append((l_prob.item(), r_prob.item()))

        if self.left:
            self.left(x, labels, sample_idx)
        if self.right:
            self.right(x, labels, sample_idx)




class DecisionTree(nn.Module):
    def __init__(self):
        super(DecisionTree, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # Build the decision tree with English names
        self.root = DecisionNode("Industrial vs Natural", judge=[[0,1,8,9],[2,3,4,5,6,7]], depth=0,
            left=DecisionNode("Sky vs Land", judge=[[0,8],[1,9]], depth=1,
                left=DecisionNode("Airplane vs Ship", judge=[[0],[8]], depth=2, left=None, right=None),
                right=DecisionNode("Car vs Truck", judge=[[1],[9]], depth=2, left=None, right=None)),
            right=DecisionNode("Others vs Quadrupeds", judge=[[2,6],[3,4,5,7]], depth=1,
                left=DecisionNode("Bird vs Frog", judge=[[2],[6]], depth=2, left=None, right=None),
                right=DecisionNode("Cat/Dog vs Deer/Horse", judge=[[3,5],[4,7]], depth=2,
                    left=DecisionNode("Cat vs Dog", judge=[[3],[5]], depth=3, left=None, right=None),
                    right=DecisionNode("Deer vs Horse", judge=[[4],[7]], depth=3, left=None, right=None)
                ),
            )
        )

    def forward(self, x):

        return self.root(x)