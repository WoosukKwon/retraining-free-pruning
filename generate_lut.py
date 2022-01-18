import argparse
import math
import os

import torch
import torch.nn as nn
from transformers import AutoConfig

from utils.timer import CPUTimer, GPUTimer


class BertMHA(nn.Module):

    def __init__(
        self,
        num_attention_heads,
        attention_head_size,
        hidden_size,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size

        self.all_head_size = num_attention_heads * attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.output = nn.Linear(self.all_head_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.output(context_layer) + hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertFFN(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.ffn1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.ffn2 = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        ffn1_output = self.ffn1(hidden_states)
        ffn1_output = self.gelu(ffn1_output)
        ffn2_output = self.ffn2(ffn1_output)
        hidden_states = ffn2_output + hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


@torch.no_grad()
def mha_lut(config, device, input_shape, num_warmup=100, num_iter=100):
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    lut = []
    for num_heads in range(1, num_attention_heads + 1):
        model = BertMHA(num_heads, attention_head_size, hidden_size)
        model = model.to(device).eval()

        x = torch.randn(input_shape).to(device)

        for _ in range(num_warmup):
            model(x)

        timelogs = []
        timer = CPUTimer(timelogs) if device == "cpu" else GPUTimer(timelogs)
        for _ in range(num_iter):
            with timer:
                model(x)

        mean_latency = sum(timer.timelogs) / num_iter
        lut.append(mean_latency)
    return lut


@torch.no_grad()
def ffn_lut(config, device, input_shape, num_warmup=10, num_iter=10):
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    lut = []
    for num_neurons in range(1, intermediate_size + 1):
        model = BertFFN(hidden_size, num_neurons)
        model = model.to(device).eval()

        x = torch.randn(input_shape).to(device)

        for _ in range(num_warmup):
            model(x)

        timelogs = []
        timer = CPUTimer(timelogs) if device == "cpu" else GPUTimer(timelogs)

        for _ in range(num_iter):
            with timer:
                model(x)

        mean_latency = sum(timer.timelogs) / num_iter
        lut.append(mean_latency)
    return lut


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_name)
    input_shape = (args.bs, args.seq_len, config.hidden_size)

    mha_latencies = mha_lut(config, args.device, input_shape)
    torch.save(mha_latencies, os.path.join(args.output_dir, "mha_lut.pt"))

    ffn_latencies = ffn_lut(config, args.device, input_shape)
    torch.save(ffn_latencies, os.path.join(args.output_dir, "ffn_lut.pt"))
