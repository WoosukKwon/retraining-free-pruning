import torch
import torch.nn as nn

from models.common.activations import ACT2FN


class BertIntermediate(nn.Module):

    def __init__(self, config, idx):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config, idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_groups = config.num_attention_heads
        self.group_size = int(config.intermediate_size / self.num_groups)

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def masked_dense(self, hidden_states, filter_mask):
        weights_per_filter = self.dense.weight.transpose(0, 1).view(
            self.num_groups,
            self.group_size,
            self.hidden_size,
        )
        hidden_states = hidden_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_groups,
            self.group_size,
        ).permute(0, 2, 1, 3)
        hidden_states = torch.matmul(hidden_states, weights_per_filter)

        filter_mask = filter_mask.view(-1, 1, 1)
        hidden_states = hidden_states * filter_mask
        hidden_states = hidden_states.sum(dim=1)
        hidden_states = hidden_states + self.dense.bias
        return hidden_states


    def forward(self, hidden_states, input_tensor, filter_mask=None):
        hidden_states = self.masked_dense(hidden_states, filter_mask=filter_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
