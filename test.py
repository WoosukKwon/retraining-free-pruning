import argparse
import os

from tqdm import tqdm
import torch
from datasets import load_metric
from accelerate import Accelerator
from transformers import AutoTokenizer

from models.bert.config import BertConfig
from models.bert.model import BertForSequenceClassification
from tools.glue import glue_dataloader


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
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tokenizer", type=str, default= None)


def main():
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_name
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        use_auth_token=None,
    )
    eval_dataloader = glue_dataloader(
        args.task_name,
        tokenizer=tokenizer,
        training=False,
        batch_size=128,
    )
    metric = load_metric("glue", args.task_name)

    config = MODELS[args.model_name][0].from_pretrained(args.ckpt_dir)
    model = MODELS[args.model_name][1].from_pretrained(args.ckpt_dir, config=config)

    accelerator = Accelerator()
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )
    model.eval()
    with torch.no_grad():
        head_masks = [torch.ones(config.num_attention_heads).cuda() for _ in range(config.num_hidden_layers)]
        filter_masks = [torch.ones(config.num_attention_heads).cuda() for _ in range(config.num_hidden_layers)]

        for batch in tqdm(eval_dataloader):
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
    print(eval_metric)


if __name__ == "__main__":
    main()
