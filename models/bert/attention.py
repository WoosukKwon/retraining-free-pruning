import math

import torch
import torch.nn as nn


class BertSelfAttention(nn.Module):

    def __init__(self, config, idx):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config, idx):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.hidden_size = config.hidden_size
        all_head_size = config.num_attention_heads * config.attention_head_size

        self.dense = nn.Linear(all_head_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def masked_dense(self, hidden_states, head_mask=None):
        weights_per_head = self.dense.weight.transpose(0, 1).view(
            self.num_attention_heads,
            self.attention_head_size,
            self.hidden_size,
        )
        hidden_states = torch.matmul(hidden_states, weights_per_head)

        head_mask = head_mask.view(-1, 1, 1)
        hidden_states = hidden_states * head_mask
        hidden_states = hidden_states.sum(dim=1)
        hidden_states = hidden_states + self.dense.bias
        return hidden_states

    def forward(self, hidden_states, input_tensor, head_mask=None):
        hidden_states = self.masked_dense(hidden_states, head_mask=head_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config, idx):
        super().__init__()
        self.self = BertSelfAttention(config, idx)
        self.output = BertSelfOutput(config, idx)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        head_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(
            self_outputs[0],
            hidden_states,
            head_mask=head_mask,
        )
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
