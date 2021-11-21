import copy
import math

from tqdm import tqdm
import torch


class MCMCFinder:

    def __init__(self, model_config, acc_predictor, efficiency_predictor, logger, ranked=False, beta=5.0):
        self.config = model_config
        self.acc_predictor = acc_predictor
        self.efficiency_predictor = efficiency_predictor
        self.logger = logger
        self.ranked = ranked

        self.beta = beta

    def random_sample_arch(self, mac_threshold):
        head_prob = torch.ones(self.config.num_attention_heads) * mac_threshold
        filter_prob = torch.ones(self.config.num_filter_groups) * mac_threshold
        head_masks = [torch.bernoulli(head_prob).cuda() for _ in range(self.config.num_hidden_layers)]
        filter_masks = [torch.bernoulli(filter_prob).cuda() for _ in range(self.config.num_hidden_layers)]
        if self.ranked:
            head_masks = [torch.sort(head_mask, descending=True)[0] for head_mask in head_masks]
            filter_masks = [torch.sort(filter_mask, descending=True)[0] for filter_mask in filter_masks]
        return {
            "head_masks": head_masks,
            "filter_masks": filter_masks,
        }

    def random_valid_sample(self, mac_threshold):
        while True:
            sample = self.random_sample_arch(mac_threshold)
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if efficiency <= mac_threshold:
                return sample

    def mutate_layer(self, config, mac_threshold):
        new_config = copy.deepcopy(config)
        head_masks = new_config["head_masks"]
        filter_masks = new_config["filter_masks"]
        head_prob = torch.ones(self.config.num_attention_heads) * mac_threshold
        filter_prob = torch.ones(self.config.num_filter_groups) * mac_threshold

        while True:
            layer_idx = torch.randint(low=0, high=self.config.num_hidden_layers, size=(1,)).item()
            mask = torch.bernoulli(head_prob).cuda()
            head_masks[layer_idx] = torch.sort(mask, descending=True)[0] if self.ranked else mask
            mask = torch.bernoulli(filter_prob).cuda()
            filter_masks[layer_idx] = torch.sort(mask, descending=True)[0] if self.ranked else mask

            mac_ratio = self.efficiency_predictor.get_efficiency(new_config)
            if mac_ratio <= mac_threshold:
                break
        return new_config, mac_ratio        

    @torch.no_grad()
    def search(self, num_iter, mac_threshold):
        num_init_candidates = int((num_iter - 1) / 100) + 1
        init_candidates = [self.random_valid_sample(mac_threshold) for _ in range(num_init_candidates)]
        num_trials = [100] * num_init_candidates
        if num_init_candidates % 100 != 0:
            num_trials[-1] = num_iter % 100

        best_acc = 0.0
        i = 0
        progress_bar = tqdm(range(num_iter))
        for init_candidate, num_trial in zip(init_candidates, num_trials):
            curr_candidate = init_candidate
            curr_acc = self.acc_predictor.predict_acc([curr_candidate])[0]
            
            for _ in range(num_trial):
                new_candidate, mac_ratio = self.mutate_layer(curr_candidate, mac_threshold)
                acc = self.acc_predictor.predict_acc([new_candidate])[0]
                if acc > best_acc:
                    best_config = new_candidate
                    best_acc = acc
                    best_mac = mac_ratio

                acceptance_prob = math.exp(self.beta * (acc - curr_acc))
                if torch.rand(1).item() < acceptance_prob:
                    curr_candidate = new_candidate
                    curr_acc = acc

                self.logger.info(f"Iter {i}  Current Acc: {curr_acc} Best Acc: {best_acc} MAC: {best_mac}")
                i += 1
                progress_bar.update(1)
        return best_config["head_masks"], best_config["filter_masks"]
