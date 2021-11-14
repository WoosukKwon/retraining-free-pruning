import argparse
import logging
import os

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoTokenizer, default_data_collator, set_seed

from models.bert.config import BertConfig
from models.bert.model import BertForSequenceClassification
from tools.glue import glue_dataloader, max_seq_length, glue_dataset, target_dev_metric
from tools.partition import partition_dataset
from search.algo.evolution import EvolutionFinder
from search.algo.random import RandomFinder
from search.predictor.accuracy import SampleAccuracyPredictor
from search.predictor.efficiency import MACPredictor


logger = logging.getLogger(__name__)

MODELS = {
    "bert-base-uncased": (BertConfig, BertForSequenceClassification),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, choices=MODELS.keys())
parser.add_argument("--task_name", type=str, required=True, choices=[
    "mrpc",
    "rte",
    "stsb",
    "sst2",
    "qnli",
    "qqp",
])
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--dataset", required=True, choices=["train", "dev"])
parser.add_argument("--sample_ratio", type=float, default=0.1)
parser.add_argument("--search_algo", required=True, choices=[
    "random",
    "evolution",
])
parser.add_argument("--num_iter", type=int, default=100)
parser.add_argument("--mac_threshold", type=float, default=0.7)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tokenizer", type=str, default= None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_dir", type=str, default=None)


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
    metric = load_metric("glue", args.task_name)

    test_dataloader = glue_dataloader(
        args.task_name,
        tokenizer,
        training=False,
        batch_size=512,
    )

    if args.dataset == "train":
        sample_dataset = glue_dataset(
            args.task_name,
            tokenizer,
            training=True,
            max_seq_len=max_seq_length(args.task_name),
        )
    else:
        sample_dataset = glue_dataset(
            args.task_name,
            tokenizer,
            training=False,
            max_seq_len=max_seq_length(args.task_name),
        )

    if args.sample_ratio == 1.0:
        sample_dataloader = DataLoader(
            sample_dataset,
            batch_size=512,
            collate_fn=default_data_collator,
            pin_memory=True,
        )
    else:
        ratios = [1 - args.sample_ratio, args.sample_ratio]
        others_sampler, sample_sampler = partition_dataset(sample_dataset, ratios)
        sample_dataloader = DataLoader(
            sample_dataset,
            sampler=sample_sampler,
            batch_size=512,
            collate_fn=default_data_collator,
            pin_memory=True,
        )
        if args.dataset == "dev":
            test_dataloader = DataLoader(
                sample_dataset,
                sampler=others_sampler,
                batch_size=512,
                collate_fn=default_data_collator,
                pin_memory=True,
            )

    avg_seq_len = get_seq_len(sample_dataloader)
    logger.info(f"Average sequence length: {avg_seq_len}")

    acc_predictor = SampleAccuracyPredictor(model, args.task_name, sample_dataloader, metric)
    mac_predictor = MACPredictor(config, avg_seq_len)

    if args.search_algo == "evolution":
        finder = EvolutionFinder(config, acc_predictor, mac_predictor, logger)
    elif args.search_algo == "random":
        finder = RandomFinder(config, acc_predictor, mac_predictor, logger)
    head_masks, filter_masks = finder.search(args.num_iter, args.mac_threshold)
    torch.save(head_masks, os.path.join(args.log_dir, "head_masks.pt"))
    torch.save(filter_masks, os.path.join(args.log_dir, "filter_masks.pt"))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            for k, v in batch.items():
                batch[k] = v.to("cuda", non_blocking=True)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                head_masks=head_masks,
                filter_masks=filter_masks,
            )
            if model.problem_type == "regression":
                predictions = outputs.logits.squeeze()
            else:
                predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )
    eval_metric = metric.compute()
    target_metric = target_dev_metric(args.task_name)
    accuracy = eval_metric[target_metric] # FIXME
    logger.info(f"Test accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
