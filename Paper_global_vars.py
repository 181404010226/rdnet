

class GlobalVars:
    def __init__(self):
        self.num_epochs = 500
        self.train_batch_size = 128
        self.test_batch_size = 1000
        self.image_probabilities = {}  

    def update_image_probabilities(self, judge, probabilities):
        for idx, probs in enumerate(probabilities):
            if idx not in self.image_probabilities:
                self.image_probabilities[idx] = {label: 1 for label in judge[0] + judge[1]}
            
            # 处理第一个概率（对应judge[0]）
            for label in judge[0]:
                self.image_probabilities[idx][label] *= probs[0]
        
            # 处理第二个概率（对应judge[1]）
            for label in judge[1]:
                self.image_probabilities[idx][label] *= probs[1]

global_vars = GlobalVars()