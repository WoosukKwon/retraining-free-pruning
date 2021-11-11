import torch

from tools.mac import compute_mac


class MACPredictor:

    def __init__(self, model_config, avg_seq_len):
        self.config = model_config
        self.seq_len = avg_seq_len
        self.baseline_mac = compute_mac(
            [model_config.num_attention_heads] * model_config.num_hidden_layers,
            [model_config.intermediate_size] * model_config.num_hidden_layers,
            avg_seq_len,
            model_config.hidden_size,
            model_config.attention_head_size,
        )

    def compute_mac_ratio(self, num_heads, num_filter_groups):
        filter_group_size = int(self.config.intermediate_size / self.config.num_filter_groups)
        num_filters = [num_groups * filter_group_size for num_groups in num_filter_groups]
        mac = compute_mac(
            num_heads,
            num_filters,
            self.seq_len,
            self.config.hidden_size,
            self.config.attention_head_size,
        )
        mac_ratio = mac / self.baseline_mac
        return mac_ratio

    def get_efficiency(self, config):
        head_masks = config["head_masks"]
        filter_masks = config["filter_masks"]
        num_heads = [mask.sum().item() for mask in head_masks]
        num_filter_groups = [mask.sum().item() for mask in filter_masks]
        mac_ratio = self.compute_mac_ratio(num_heads, num_filter_groups)
        return mac_ratio
