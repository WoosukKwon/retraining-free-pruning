import argparse
import logging
import os

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoTokenizer, set_seed, DataCollatorWithPadding

from models.bert.config import BertConfig
from models.bert.model import BertForQuestionAnswering
from tools.partition import partition_dataset
from tools.squad import squad_dataset
from search.algo.evolution import EvolutionFinder
from search.algo.random import RandomFinder
from search.predictor.accuracy import SampleAccuracyPredictor
from search.predictor.efficiency import MACPredictor


logger = logging.getLogger(__name__)

MODELS = {
    "bert-base-uncased": (BertConfig, BertForQuestionAnswering),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, choices=MODELS.keys())
parser.add_argument("--task_name", type=str, required=True, choices=[
    "squad",
    "squad_v2",
])
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--dataset", required=True, choices=["train", "dev"])
parser.add_argument("--sample_ratio", type=float, default=1.0)
parser.add_argument("--search_algo", required=True, choices=[
    "random",
    "evolution",
])
parser.add_argument("--ranked", action="store_true")
parser.add_argument("--num_iter", type=int, default=100)
parser.add_argument("--mac_threshold", type=float, default=0.6)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tokenizer", type=str, default= None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_dir", type=str, default=None)


@torch.no_grad()
def get_seq_len(dataloader):
    val = 0.0
    cnt = 0
    for batch in dataloader:
        seq_len = batch["attention_mask"].sum()
        val += seq_len.item()
        cnt += batch["attention_mask"].shape[0]
    avg_seq_len = val / cnt
    return avg_seq_len


def main():
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_name
    if args.log_dir is None:
        args.log_dir = os.path.join(
            "logs",
            args.model_name,
            args.task_name,
            args.search_algo,
            f"mac_{args.mac_threshold}",
            args.dataset,
            f"sample_{args.sample_ratio}",
            "ranked" if args.ranked else "unranked",
            f"num_iter_{args.num_iter}",
            f"seed_{args.seed}",
        )
    os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.log_dir, "log.txt")),
        ],
    )
    logger.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")
    
    config = MODELS[args.model_name][0].from_pretrained(args.ckpt_dir)
    model = MODELS[args.model_name][1].from_pretrained(args.ckpt_dir, config=config)
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        use_auth_token=None,
    )
    metric = load_metric(args.task_name)

    collate_fn = DataCollatorWithPadding(tokenizer)
    if args.dataset == "train" or args.sample_ratio != 1.0:
        assert False, "Not supported" # FIXME

    # dev set
    sample_dataset, sample_examples = squad_dataset(
        args.task_name,
        tokenizer,
        training=False,
        max_seq_len=384,
        pad_to_max=False,
    )
    sample_dataloader = DataLoader(
        sample_dataset.remove_columns(["example_id", "offset_mapping"]),
        batch_size=128,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=False,
    )

    avg_seq_len = get_seq_len(sample_dataloader)
    logger.info(f"Sample average sequence length: {avg_seq_len}")

    acc_predictor = SampleAccuracyPredictor(
        model,
        args.task_name,
        sample_dataloader,
        metric,
        eval_dataset=sample_dataset,
        eval_examples=sample_examples,
    )
    mac_predictor = MACPredictor(config, avg_seq_len)

    if args.search_algo == "evolution":
        finder = EvolutionFinder(config, acc_predictor, mac_predictor, logger, ranked=args.ranked)
    elif args.search_algo == "random":
        finder = RandomFinder(config, acc_predictor, mac_predictor, logger, ranked=args.ranked)

    head_masks, filter_masks = finder.search(args.num_iter, args.mac_threshold)
    torch.save(head_masks, os.path.join(args.log_dir, "head_masks.pt"))
    torch.save(filter_masks, os.path.join(args.log_dir, "filter_masks.pt"))
    mac_ratio = mac_predictor.get_efficiency({
        "head_masks": head_masks,
        "filter_masks": filter_masks,
    })
    logger.info(f"Best config MAC: {mac_ratio * 100.0:.2f} %")

    # TODO: Measure test accuracy

if __name__ == "__main__":
    main()
