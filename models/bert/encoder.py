import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from models.bert.attention import BertAttention
from models.common.token_pruner import TokenPruner
from models.bert.ffn import BertIntermediate, BertOutput


class BertLayer(nn.Module):

    def __init__(self, config, idx):
        super().__init__()
        self.layer_idx = idx

        self.attention = BertAttention(config, idx)
        self.token_pruner = TokenPruner(config, idx) if config.prune_tokens else None
        self.intermediate = BertIntermediate(config, idx)
        self.output = BertOutput(config, idx)

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        filter_mask=None,
        temp=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=True,
            head_mask=head_mask,
        )
        attention_output = self_attention_outputs[0]
        attention_probs = self_attention_outputs[1]

        if self.token_pruner is not None:
            attention_output, attention_mask = self.token_pruner(
                attention_output,
                attention_probs,
                head_mask,
                attention_mask,
                temp,
            )

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, filter_mask=filter_mask)
        return layer_output, attention_mask


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_masks=None,
        filter_masks=None,
        temp=None,
    ):
        for i, layer_module in enumerate(self.layer):
            hidden_states, attention_mask = layer_module(
                hidden_states,
                attention_mask,
                head_mask=head_masks[i],
                filter_mask=filter_masks[i],
                temp=temp,
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )
