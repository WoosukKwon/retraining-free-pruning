import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoTokenizer, default_data_collator, set_seed

from models.bert.config import BertConfig
from models.bert.model import BertForSequenceClassification
from tools.glue import max_seq_length, glue_dataset


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
parser.add_argument("--sample_size", type=int, default=256)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tokenizer", type=str, default= None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_dir", type=str, default=None)


def sample_data(task_name, tokenizer, batch_size):
    max_seq_len = max_seq_length(task_name)
    dataset = glue_dataset(
        task_name,
        tokenizer,
        training=False,
        max_seq_len=max_seq_len,
        pad_to_max=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    sample_batch = next(iter(dataloader))
    del dataloader
    return sample_batch


@torch.no_grad()
def evaluate(model, sample_batch, head_masks, filter_masks, metric):
    model.eval()
    outputs = model(
        input_ids=sample_batch["input_ids"],
        attention_mask=sample_batch["attention_mask"],
        head_masks=head_masks,
        filter_masks=filter_masks,
    )
    if model.problem_type == "regression":
        predictions = outputs.logits.squeeze()
    else:
        predictions = outputs.logits.argmax(dim=-1)
    metric.add_batch(
        predictions=predictions,
        references=sample_batch["labels"],
    )
    eval_metric = metric.compute()
    accuracy = eval_metric["accuracy"] # FIXME
    return accuracy * 100.0


def main():
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_name
    if args.log_dir is None:
        args.log_dir = os.path.join(
            "logs",
            args.model_name,
            args.task_name,
            f"sample_{args.sample_size}",
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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")
    
    config = MODELS[args.model_name][0].from_pretrained(args.ckpt_dir)
    model = MODELS[args.model_name][1].from_pretrained(args.ckpt_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        use_auth_token=None,
    )

    sample_batch = sample_data(args.task_name, tokenizer, args.sample_size)
    metric = load_metric("glue", args.task_name)

    model = model.cuda()
    sample_batch["input_ids"] = sample_batch["input_ids"].cuda()
    sample_batch["attention_mask"] = sample_batch["attention_mask"].cuda()
    sample_batch["labels"] = sample_batch["labels"].cuda()
    
    head_masks = [torch.ones(config.num_attention_heads).cuda() for _ in range(config.num_hidden_layers)]
    filter_masks = [torch.ones(config.num_attention_heads).cuda() for _ in range(config.num_hidden_layers)]

    accuracy = evaluate(model, sample_batch, head_masks, filter_masks, metric)
    print(accuracy)


if __name__ == "__main__":
    main()
