import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoTokenizer, set_seed, DataCollatorWithPadding

from models.bert.config import BertConfig
from models.bert.model import BertForQuestionAnswering, BertForSequenceClassification
from tools.glue import glue_dataloader, max_seq_length, glue_dataset
from tools.squad import squad_dataset
from tools.partition import partition_dataset
from tools.importance import importance_by_gradient
from search.algo.random import RandomFinder
from search.algo.evolution import EvolutionFinder
from search.algo.mcmc import MCMCFinder
from search.algo.ilp import ILPFinder
from search.predictor.accuracy import SampleAccuracyPredictor
from search.predictor.efficiency import MACPredictor


logger = logging.getLogger(__name__)

MODELS = {
    "bert-base-uncased": (BertConfig, BertForSequenceClassification, BertForQuestionAnswering),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, choices=MODELS.keys())
parser.add_argument("--task_name", type=str, required=True, choices=[
    "mnli",
    "mrpc",
    "rte",
    "stsb",
    "sst2",
    "qnli",
    "qqp",
    "squad",
    "squad_v2",
])
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--dataset", required=True, choices=["train", "dev"])
parser.add_argument("--sample_ratio", type=float, default=1.0)
parser.add_argument("--search_algo", required=True, choices=[
    "random",
    "evolution",
    "mcmc",
    "ilp",
])
parser.add_argument("--ranked", action="store_true")
parser.add_argument("--num_iter", type=int, default=100)
parser.add_argument("--mac_threshold", type=float, default=0.6)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--comment", type=str, default=None)


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
    if args.comment is not None:
        logger.info(args.comment)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")

    is_squad = "squad" in args.task_name

    config = MODELS[args.model_name][0].from_pretrained(args.ckpt_dir)
    if is_squad:
        model = MODELS[args.model_name][2].from_pretrained(args.ckpt_dir, config=config)
    else:
        model = MODELS[args.model_name][1].from_pretrained(args.ckpt_dir, config=config)
    model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        use_auth_token=None,
    )
    metric = load_metric(args.task_name) if is_squad else load_metric("glue", args.task_name)

    collate_fn = DataCollatorWithPadding(tokenizer)
    if is_squad:
        if args.dataset == "train":
            sample_dataset = squad_dataset(
                args.task_name,
                tokenizer,
                training=True,
                max_seq_len=384,
                pad_to_max=False,
            )
        else:
            raise NotImplementedError("Use train set for the SQuAD datasets") # FIXME
    else:
        sample_dataset = glue_dataset(
            args.task_name,
            tokenizer,
            training=args.dataset == "train",
            max_seq_len=max_seq_length(args.task_name),
            pad_to_max=False,
        )

    sample_batch_size = 128 if is_squad else 512
    if args.sample_ratio == 1.0:
        ratios = [1.0]
    else:
        ratios = [1 - args.sample_ratio, args.sample_ratio]
    samplers = partition_dataset(sample_dataset, ratios)
    sample_sampler = samplers[-1]
    sample_dataloader = DataLoader(
        sample_dataset,
        sampler=sample_sampler,
        batch_size=sample_batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    logger.info(f"# search examples: {len(sample_sampler)}")

    if args.dataset == "dev":
        test_sampler = samplers[0]
        test_dataloader = DataLoader(
            sample_dataset,
            sampler=test_sampler,
            batch_size=sample_batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        logger.info(f"# test examples: {len(test_sampler)}")
    else:
        if is_squad:
            squad_dev_set, squad_dev_examples = squad_dataset(
                args.task_name,
                tokenizer,
                training=False,
                max_seq_len=384,
                pad_to_max=False,
            )
            test_dataloader = DataLoader(
                squad_dev_set.remove_columns(["example_id", "offset_mapping"]),
                batch_size=sample_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
        else:
            test_dataloader = glue_dataloader(
                args.task_name,
                tokenizer,
                training=False,
                batch_size=sample_batch_size,
                pad_to_max=False,
            )
        logger.info(f"# test examples: {len(test_dataloader)}")

    avg_seq_len = get_seq_len(sample_dataloader)
    logger.info(f"Sample average sequence length: {avg_seq_len}")

    acc_predictor = SampleAccuracyPredictor(model, args.task_name, sample_dataloader, metric)
    mac_predictor = MACPredictor(config, avg_seq_len)

    full_network_config = {
        "head_masks": [torch.ones(config.num_attention_heads).cuda() for _ in range(config.num_hidden_layers)],
        "filter_masks": [torch.ones(config.num_filter_groups).cuda() for _ in range(config.num_hidden_layers)],
    }
    if is_squad:
        baseline_loss = acc_predictor.predict_loss([full_network_config])[0]
        logger.info(f"Full network loss on samples: {baseline_loss:.4f}")
    else:
        baseline_acc = acc_predictor.predict_acc([full_network_config])[0]
        logger.info(f"Full network acc on samples: {baseline_acc:.4f}")

    if args.search_algo == "ilp":
        sample_dataloader = DataLoader(
            sample_dataset,
            sampler=sample_sampler,
            batch_size=12 if is_squad else 32,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        head_importance, filter_importance = importance_by_gradient(
            model,
            config,
            sample_dataloader,
            absolute=True,
        )

    if args.search_algo == "evolution":
        finder = EvolutionFinder(config, acc_predictor, mac_predictor, logger, ranked=args.ranked)
    elif args.search_algo == "random":
        finder = RandomFinder(config, acc_predictor, mac_predictor, logger, ranked=args.ranked)
    elif args.search_algo == "mcmc":
        finder = MCMCFinder(config, acc_predictor, mac_predictor, logger, ranked=args.ranked)
    elif args.search_algo == "ilp":
        finder = ILPFinder(
            config,
            acc_predictor,
            mac_predictor,
            logger,
            head_importance,
            filter_importance,
            use_loss=is_squad,
        )

    best_config = finder.search(args.num_iter, args.mac_threshold)
    mac_ratio = mac_predictor.get_efficiency(best_config)
    logger.info(f"Best config MAC: {mac_ratio * 100.0:.2f} %")

    head_masks = best_config["head_masks"]
    filter_masks = best_config["filter_masks"]
    torch.save(head_masks, os.path.join(args.log_dir, "head_masks.pt"))
    torch.save(filter_masks, os.path.join(args.log_dir, "filter_masks.pt"))

    if is_squad:
        test_acc_predictor = SampleAccuracyPredictor(
            model,
            args.task_name,
            test_dataloader,
            metric,
            squad_dev_set,
            squad_dev_examples,
        )
    else:
        test_acc_predictor = SampleAccuracyPredictor(
            model,
            args.task_name,
            test_dataloader,
            metric,
        )
    test_acc = test_acc_predictor.predict_acc([best_config])[0]
    logger.info(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
