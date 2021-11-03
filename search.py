import argparse
import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from accelerate import Accelerator
from transformers import AutoTokenizer, default_data_collator, set_seed

from models.bert.config import BertConfig
from models.bert.model import BertForSequenceClassification
from tools.glue import max_seq_length, glue_dataset


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
def evaluate(model, sample_batch, head_mask, filter_mask, metric):
    model.eval()
    outputs = model(
        input_ids=sample_batch["input_ids"],
        attention_mask=sample_batch["attention_mask"],
        head_mask=head_mask,
        filter_mask=filter_mask,
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
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    
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
    head_mask = torch.ones(config.num_attention_heads).cuda()
    filter_mask = torch.ones(config.num_attention_heads).cuda()
    sample_batch["input_ids"] = sample_batch["input_ids"].cuda()
    sample_batch["attention_mask"] = sample_batch["attention_mask"].cuda()
    sample_batch["labels"] = sample_batch["labels"].cuda()

    accuracy = evaluate(model, sample_batch, head_mask, filter_mask, metric)
    print(accuracy)


if __name__ == "__main__":
    main()