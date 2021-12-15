import argparse

import torch

from models.bert.config import BertConfig
from models.bert.attention import BertAttention
from models.bert.ffn import BertIntermediate, BertOutput
from tools.timer import CPUTimer, GPUTimer


@torch.no_grad()
def mha_lut(config, device, input_shape):
    lut = []
    for num_heads in range(1, 13):
        config.num_attention_heads = num_heads

        model = BertAttention(config, 0).to(device).eval()
        x = torch.randn(input_shape).to(device)
        attention_mask = torch.ones(input_shape[0], 1, input_shape[1], input_shape[1]).to(device)
        head_mask = torch.ones(num_heads).to(device)

        timelogs = []
        timer = CPUTimer(timelogs) if device == "cpu" else GPUTimer(timelogs)

        NUM_WARMUP = 100
        for _ in range(NUM_WARMUP):
            model(x, attention_mask, head_mask=head_mask)

        NUM_ITER = 100
        for _ in range(NUM_ITER):
            with timer:
                model(x, attention_mask, head_mask=head_mask)
        mean = sum(timer.timelogs) / NUM_ITER
        lut.append(mean)
    return lut


@torch.no_grad()
def ffn_lut(config, device, input_shape):
    lut = []
    for num_filters in range(1, 3073):
        config.intermediate_size = num_filters
        config.num_filter_groups = num_filters

        ffn1 = BertIntermediate(config, 0).to(device).eval()
        ffn2 = BertOutput(config, 0).to(device).eval()
        x = torch.randn(input_shape).to(device)
        filter_mask = torch.ones(num_filters).to(device)

        timelogs = []
        timer = CPUTimer(timelogs) if device == "cpu" else GPUTimer(timelogs)

        NUM_WARMUP = 10
        for _ in range(NUM_WARMUP):
            y = ffn1(x)
            ffn2(y, x, filter_mask=filter_mask)

        NUM_ITER = 10
        for _ in range(NUM_ITER):
            with timer:
                y = ffn1(x)
                ffn2(y, x, filter_mask=filter_mask)
        mean = sum(timer.timelogs) / NUM_ITER
        lut.append(mean)
    return lut


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    args = parser.parse_args()

    config = BertConfig()
    input_shape = (args.bs, args.seq_len, config.hidden_size)
    mha_latencies = mha_lut(config, args.device, input_shape)
    torch.save(mha_latencies, f"MHA_{args.device}_bs{args.bs}_len{args.seq_len}_lut.pt")
    ffn_latencies = ffn_lut(config, args.device, input_shape)
    torch.save(ffn_latencies, f"FFN_{args.device}_bs{args.bs}_len{args.seq_len}_lut.pt")
