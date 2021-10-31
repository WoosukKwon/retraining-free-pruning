import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from models.bert.attention import BertAttention
from models.bert.ffn import BertIntermediate, BertOutput


class BertLayer(nn.Module):

    def __init__(self, config, idx):
        super().__init__()
        self.layer_idx = idx

        self.attention = BertAttention(config, idx)
        self.intermediate = BertIntermediate(config, idx)
        self.output = BertOutput(config, idx)

    def forward(
        self,
        hidden_states,
        attention_mask,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=False,
        )
        attention_output = self_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
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
    ):
        for i, layer_module in enumerate(self.layer):
            hidden_states, attention_mask = layer_module(
                hidden_states,
                attention_mask,
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )
