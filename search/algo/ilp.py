from tqdm import tqdm
import torch


class ILPFinder:

    def __init__(
        self,
        model_config,
        acc_predictor,
        efficiency_predictor,
        logger,
        head_importance,
        filter_importance,
        use_loss=False,
    ):
        self.config = model_config
        self.acc_predictor = acc_predictor
        self.efficiency_predictor = efficiency_predictor
        self.logger = logger

        self.head_importance = head_importance
        self.filter_importance = filter_importance
        self.use_loss = use_loss

    def search(self, num_iter, mac_threshold):
        head_importance = self.head_importance.view(-1)
        filter_importance = torch.cat(self.filter_importance, dim=-1)

        sorted_head_importance, sorted_head_indicies = head_importance.sort(descending=True)
        sorted_filter_importance, sorted_filter_indicies = filter_importance.sort(descending=True)

        best_result = 0
        best_mac = 0
        best_config = None
        num_total_heads = self.config.num_attention_heads * self.config.num_hidden_layers
        for num_heads in tqdm(range(num_total_heads + 1)):
            if num_heads == 0:
                head_masks = [torch.zeros(self.config.num_attention_heads).cuda()] * self.config.num_hidden_layers
                num_heads_per_layer = [0] * self.config.num_hidden_layers
            else:
                head_threshold = sorted_head_importance[num_heads - 1]
                head_masks = (head_importance >= head_threshold).view(self.config.num_hidden_layers, -1)
                num_heads_per_layer = [mask.sum().item() for mask in head_masks]

            num_filter_groups = self.efficiency_predictor.get_max_num_filter_groups(num_heads_per_layer, mac_threshold)
            filter_threshold = sorted_filter_importance[num_filter_groups - 1] # NOTE: no clustering!
            filter_masks = (filter_importance >= filter_threshold).view(self.config.num_hidden_layers, -1)
            num_filters_per_layer = [mask.sum().item() for mask in filter_masks]

            if num_heads == 0:
                total_importance = sorted_filter_importance[:num_filter_groups].sum()
            else:
                total_importance = sorted_head_importance[:num_heads].sum() + sorted_filter_importance[:num_filter_groups].sum()
            config = {
                "head_masks": head_masks,
                "filter_masks": filter_masks,
            }
            mac_ratio = self.efficiency_predictor.get_efficiency(config)
            self.logger.info(
                f"Num heads: {num_heads_per_layer} "
                f"Num filters: {num_filters_per_layer} "
                f"Total importance {total_importance:.4f} "
                f"MAC: {mac_ratio:.4f} "
            )
            if mac_ratio > mac_threshold:
                self.logger.info(f"MAC: {mac_ratio:.4f} BUG: the configuration must not exceed the threshold")

            if total_importance > best_result:
                best_result = total_importance
                best_mac = mac_ratio
                best_config = config
            self.logger.info(f"Best Score: {best_result} MAC: {best_mac}")

        return best_config
