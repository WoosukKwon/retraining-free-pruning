import argparse
import os

import torch


@torch.no_grad()
def lookup_latency(lut, mask):
    n = mask.sum()
    if n == 0:
        return 0
    else:
        return lut[n - 1]


def compute_total_latency(mha_lut, ffn_lut, head_masks, filter_masks):
    total = 0
    for head_mask in head_masks:
        total += lookup_latency(mha_lut, head_mask)
    for filter_mask in filter_masks:
        total += lookup_latency(ffn_lut, filter_mask)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mha_lut", type=str, required=True)
    parser.add_argument("--ffn_lut", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    args = parser.parse_args()

    mha_lut = torch.load(args.mha_lut)
    ffn_lut = torch.load(args.ffn_lut)
    head_masks = torch.load(os.path.join(args.mask_dir, "head_masks.pt"))
    filter_masks = torch.load(os.path.join(args.mask_dir, "filter_masks.pt"))

    latency = compute_total_latency(mha_lut, ffn_lut, head_masks, filter_masks)
    baseline = compute_total_latency(
        mha_lut,
        ffn_lut,
        torch.ones_like(head_masks),
        torch.ones_like(filter_masks),
    )
    print(f"LAT: {latency:.4f}, Baseline: {baseline:4f}, Speedup: {baseline / latency:.2f}")
