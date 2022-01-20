import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)
from models.wav2vec2.model_wav2vec2 import Wav2Vec2ForCTC

from dataset.librispeech import Orthography, DataCollatorCTCWithPadding, load_shard_dataset
from efficiency.mac import compute_mask_mac
from efficiency.latency import estimate_latency
from prune.fisher import collect_mask_grads
from prune.search import search_mac, search_latency
from prune.rearrange import rearrange_mask
from prune.merge import merge_neurons
from prune.rescale import rescale_mask
from evaluate.nlp import test_accuracy
from evaluate.librispeech import test_wer


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='facebook/wav2vec2-base-960h')
parser.add_argument("--task_name", type=str, choices=['librispeech'], default='librispeech')
parser.add_argument("--train_dataset_path", type=str, required=True)
parser.add_argument("--valid_dataset_path", type=str, required=True)
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument("--threshold", type=float, default=0)
parser.add_argument("--metric", type=str, choices=[
    "mac",
    "latency",
], default="mac")
parser.add_argument("--constraint", type=float, default=0.5,
    help="MAC/latency constraint relative to the origin model",
)
parser.add_argument("--mha_lut", type=str, default=None)
parser.add_argument("--ffn_lut", type=str, default=None)
parser.add_argument("--num_samples", type=int, default=2048)
parser.add_argument("--seed", type=int, default=0)


def main():
    args = parser.parse_args()

    # Create the output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs",
            args.model_name,
            args.task_name,
            f"{args.metric}_{args.constraint}",
            f"threshold_{args.threshold}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Initiate the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
        ],
    )
    logger.info(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")

    orthography = Orthography.from_name(args.task_name)
    processor = orthography.create_processor(args.model_name)

    training_dataset = load_shard_dataset(args.train_dataset_path)
    valid_dataset = load_shard_dataset(args.valid_dataset_path) 

    collate_fn = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        cache_dir=None,
        gradient_checkpointing=False,
        vocab_size=len(processor.tokenizer),
    )

    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), args.num_samples).tolist(),
    )

    sample_batch_size = 16
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    model = model.cuda()
    model.eval()
    config = model.wav2vec2.config

    for param in model.parameters():
        param.requires_grad_(False)

    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).cuda()
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).cuda()

    # TODO, this code is copied from run_nlp.py
    '''
    # Search the optimal mask
    head_grads, neuron_grads = collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        sample_dataloader,
    )
    teacher_constraint = 0.9
    if args.metric == "mac":
        teacher_head_mask, teacher_neuron_mask = search_mac(
            config,
            head_grads,
            neuron_grads,
            seq_len,
            teacher_constraint,
        )
        head_mask, neuron_mask = search_mac(
            config,
            head_grads,
            neuron_grads,
            seq_len,
            args.constraint,
        )
        pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, config.hidden_size)
        logger.info(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")
    elif args.metric == "latency":
        mha_lut = torch.load(args.mha_lut)
        ffn_lut = torch.load(args.ffn_lut)
        teacher_head_mask, teacher_neuron_mask = search_latency(
            config,
            head_grads,
            neuron_grads,
            teacher_constraint,
            mha_lut,
            ffn_lut,
        )
        head_mask, neuron_mask = search_latency(
            config,
            head_grads,
            neuron_grads,
            args.constraint,
            mha_lut,
            ffn_lut,
        )
        orig_latency = estimate_latency(mha_lut, ffn_lut, full_head_mask, full_neuron_mask)
        pruned_latency = estimate_latency(mha_lut, ffn_lut, head_mask, neuron_mask)
        logger.info(f"Full Model Latency: {orig_latency:.2f} ms")
        logger.info(f"Pruned Model Latency: {pruned_latency:.2f} ms ({pruned_latency / orig_latency * 100.0:.2f} %)")

    # Rearrange the mask
    head_mask = rearrange_mask(head_mask, head_grads)
    neuron_mask = rearrange_mask(neuron_mask, neuron_grads)

    # Merge pruned neurons into remaining neurons
    if args.threshold > 0:
        merge_neurons(model, head_mask, neuron_mask, args.threshold, sample_dataloader)

    # FIXME
    # Rescale the mask by solving a least squares problem
    head_mask, neuron_mask = rescale_mask(
        model,
        config,
        teacher_head_mask,
        teacher_neuron_mask,
        head_mask,
        neuron_mask,
        sample_dataloader,
        classification_task=not IS_SQUAD,
    )
    '''

    head_mask, neuron_mask = None, None # TODO remove this later

    wer = test_wer(
        model,
        head_mask=head_mask,
        neuron_mask=neuron_mask,
        dataset=valid_dataset,
        collate_fn=collate_fn,
        processor=processor,
    )

    logger.info(f"Test WER: {wer:.4f}")

    # Save the masks
    torch.save(head_mask, os.path.join(args.output_dir, "head_mask.pt"))
    torch.save(neuron_mask, os.path.join(args.output_dir, "neuron_mask.pt"))


if __name__ == "__main__":
    main()
