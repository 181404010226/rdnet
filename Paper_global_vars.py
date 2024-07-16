import torch

class GlobalVars:
    def __init__(self):
        self.num_epochs = 500
        self.train_batch_size = 8
        self.test_batch_size = 1000
        self.image_probabilities = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_image_probabilities(self, num_labels):
        self.image_probabilities = torch.ones(self.train_batch_size, num_labels,device=self.device)

    def update_image_probabilities(self, judge, probabilities):
        # 确保 image_probabilities 已初始化
        if self.image_probabilities is None:
            raise ValueError("image_probabilities not initialized. Call initialize_image_probabilities first.")

        # 处理第一个概率（对应judge[0]）
        self.image_probabilities[:, judge[0]] *= probabilities[:, 0].unsqueeze(1)
        
        # 处理第二个概率（对应judge[1]）
        self.image_probabilities[:, judge[1]] *= probabilities[:, 1].unsqueeze(1)

global_vars = GlobalVars()