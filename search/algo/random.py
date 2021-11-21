from tqdm import tqdm
import torch


class RandomFinder:

    def __init__(self, model_config, acc_predictor, efficiency_predictor, logger, ranked=False):
        self.config = model_config
        self.acc_predictor = acc_predictor
        self.efficiency_predictor = efficiency_predictor
        self.logger = logger
        self.ranked = ranked

    @torch.no_grad()
    def search(self, num_iter, mac_threshold):
        head_probs = [
            torch.ones(self.config.num_attention_heads) * mac_threshold
            for _ in range(self.config.num_hidden_layers)
        ]
        filter_probs = [
            torch.ones(self.config.num_filter_groups) * mac_threshold
            for _ in range(self.config.num_hidden_layers)
        ]

        best_acc = 0
        best_mac = 0
        best_config = None
        i = 0
        progress_bar = tqdm(range(num_iter))
        while i < num_iter:
            head_masks = [torch.bernoulli(prob).cuda() for prob in head_probs]
            filter_masks = [torch.bernoulli(prob).cuda() for prob in filter_probs]
            if self.ranked:
                head_masks = [torch.sort(head_mask, descending=True)[0] for head_mask in head_masks]
                filter_masks = [torch.sort(filter_mask, descending=True)[0] for filter_mask in filter_masks]

            config = {
                "head_masks": head_masks,
                "filter_masks": filter_masks,
            }
            mac_ratio = self.efficiency_predictor.get_efficiency(config)
            if mac_ratio > mac_threshold:
                continue

            acc = self.acc_predictor.predict_acc([config])[0]
            if acc > best_acc:
                best_acc = acc
                best_mac = mac_ratio
                best_config = config
            progress_bar.set_postfix({"acc": best_acc})
            self.logger.info(f"Iter {i} Acc: {best_acc} MAC: {best_mac}")
            i += 1
            progress_bar.update(1)
        return best_config["head_masks"], best_config["filter_masks"]
