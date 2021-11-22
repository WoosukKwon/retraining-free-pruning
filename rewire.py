import argparse
import os

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import set_seed, AutoTokenizer, DataCollatorWithPadding

from models.bert.config import BertConfig
from models.bert.model import BertForQuestionAnswering, BertForSequenceClassification
from tools.glue import glue_dataloader
from tools.squad import squad_dataset
from tools.rewire import rewire_by_gradient


# TODO: Add more models
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, choices=[
    "mrpc",
    "rte",
    "stsb",
    "sst2",
    "qnli",
    "qqp",
    "mnli",
    "squad",
    "squad_v2",
])
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
parser.add_argument("--seed", type=int, default=0)


def main():
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    is_squad = args.task_name in ["squad", "squad_v2"]

    config = BertConfig.from_pretrained(args.ckpt_dir)
    if is_squad:
        model = BertForQuestionAnswering.from_pretrained(args.ckpt_dir, config=config)
    else:
        model = BertForSequenceClassification.from_pretrained(args.ckpt_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if is_squad:
        train_dataset = squad_dataset(args.task_name, tokenizer, training=True, max_seq_len=384, pad_to_max=False)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=12,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
        )
    else:
        train_dataloader = glue_dataloader(
            args.task_name,
            tokenizer=tokenizer,
            training=True,
            batch_size=32,
        )

    accelerator = Accelerator()
    model, train_dataloader = accelerator.prepare(
        model, train_dataloader
    )
    rewire_by_gradient(model, train_dataloader, absolute=True)

    torch.save(
        model.state_dict(),
        os.path.join(args.ckpt_dir, "pytorch_model.bin"),
    )

if __name__ == "__main__":
    main()
