class GlobalVars:
    def __init__(self):
        self.node_stats = {}
        self.num_epochs = 200
        self.train_batch_size = 32
        self.test_batch_size = 1000
        self.image_probabilities = {}  

    def reset_stats(self):
        for key in self.node_stats:
            self.node_stats[key] = {"correct": 0, "total": 0}

    def update_stats(self, node_name, correct, total):
        if node_name not in self.node_stats:
            self.node_stats[node_name] = {"correct": 0, "total": 0}
        self.node_stats[node_name]["correct"] += correct
        self.node_stats[node_name]["total"] += total

    def update_image_probabilities(self, images, judge, probabilities):
        for img, probs in zip(images, probabilities):
            img_key = tuple(img.flatten().tolist())  # Convert image tensor to tuple for hashing
            if img_key not in self.image_probabilities:
                self.image_probabilities[img_key] = {}
            for class_idx, prob in enumerate(probs):
                self.image_probabilities[img_key][judge[class_idx]] =  prob.cpu().detach() 


    def get_accuracy(self, node_name):
        if node_name in self.node_stats:
            stats = self.node_stats[node_name]
            return stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        return 0

    def print_all_stats(self):
        for node_name, stats in self.node_stats.items():
            accuracy = self.get_accuracy(node_name)
            print(f"{node_name}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

global_vars = GlobalVars()