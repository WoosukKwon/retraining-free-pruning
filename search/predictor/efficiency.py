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

        self.filter_group_size = int(self.config.intermediate_size / self.config.num_filter_groups)
        self.per_head_mac = 2 * self.config.attention_head_size * self.seq_len * (self.seq_len + 2 * self.config.hidden_size)
        self.per_filter_group_mac = 2 * self.config.hidden_size * self.seq_len * self.filter_group_size

    def compute_mac_ratio(self, num_heads, num_filter_groups):
        num_filters = [num_groups * self.filter_group_size for num_groups in num_filter_groups]
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

    def get_max_num_heads(self, num_filter_groups, mac_threshold):
        max_mac = mac_threshold * self.baseline_mac
        filter_mac = sum(num_filter_groups) * self.per_filter_group_mac
        max_head_mac = max_mac - filter_mac
        max_num_heads = int(max_head_mac / self.per_head_mac)
        max_num_heads = min(max_num_heads, self.config.num_hidden_layers * self.config.num_attention_heads)
        return max_num_heads

    def get_max_num_filter_groups(self, num_heads, mac_threshold):
        max_mac = mac_threshold * self.baseline_mac
        head_mac = sum(num_heads) * self.per_head_mac
        max_filter_group_mac = max_mac - head_mac
        max_num_filter_groups = int(max_filter_group_mac / self.per_filter_group_mac)
        max_num_filter_groups = min(max_num_filter_groups, self.config.num_hidden_layers * self.config.num_filter_groups)
        return max_num_filter_groups
