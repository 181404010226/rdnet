import torch

import torch
import torch.nn.functional as F

class GlobalVars:
    def __init__(self):
        self.num_epochs = 300
        self.train_batch_size = 64
        self.test_batch_size = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_image_probabilities = None

    def initialize_image_probabilities(self, batch_size):
        self.log_image_probabilities = torch.ones(batch_size, 10, device=self.device)

    def update_image_probabilities(self, judge, outputs):
        outputs = torch.sigmoid(outputs)
        self.log_image_probabilities[:, judge[0]] *= outputs[:, 0].unsqueeze(1)
        self.log_image_probabilities[:, judge[1]] *= outputs[:, 1].unsqueeze(1)


global_vars = GlobalVars()  # 假设有10个标签