from tqdm import tqdm
import torch


class RandomFinder:

    def __init__(self, model_config, acc_predictor, efficiency_predictor):
        self.config = model_config
        self.acc_predictor = acc_predictor
        self.efficiency_predictor = efficiency_predictor

    def search(self, num_iter, mac_threshold):
        head_probs = [
            torch.ones(self.config.num_attention_heads) * mac_threshold
            for _ in range(self.config.num_hidden_layers)
        ]
        filter_probs = [
            torch.ones(self.config.num_attention_heads) * mac_threshold
            for _ in range(self.config.num_hidden_layers)
        ]

        best_acc = 0
        i = 0
        progress_bar = tqdm(range(num_iter))
        while i < num_iter:
            head_masks = [torch.bernoulli(prob).cuda() for prob in head_probs]
            filter_masks = [torch.bernoulli(prob).cuda() for prob in filter_probs]
            config = {
                "head_masks": head_masks,
                "filter_masks": filter_masks,
            }
            mac_ratio = self.efficiency_predictor.get_efficiency(config)
            if mac_ratio > mac_threshold:
                continue

            acc = self.acc_predictor.predict_acc([config])
            if acc > best_acc:
                best_acc = acc
            progress_bar.set_postfix({"acc": best_acc})
            i += 1
            progress_bar.update(1)
        return config
