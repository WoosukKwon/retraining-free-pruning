import argparse
import logging
import os

from tqdm import tqdm
import numpy as np
import torch
from datasets import load_metric
from transformers import AutoTokenizer, set_seed
from transformers import AdamW, get_scheduler

from models.bert.config import BertConfig
from models.bert.model import BertForSequenceClassification
from tools.glue import glue_dataloader
from tools.mac import compute_mac
from tools.meter import AverageMeter


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
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--reg_lambda", type=float, default=0.5)
parser.add_argument("--init_temp", type=float, default=1.0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default=None)


def schedule_temp(
    step: int,
    total_step: int,
    init_temp: float,
):
    temp = init_temp * (1 - step / total_step)
    return temp


def config_to_mac(model_config, head_config, filter_config, avg_seq_len):
    filter_group_size = int(model_config.intermediate_size / model_config.num_filter_groups)
    mac = compute_mac(
        head_config,
        [num_filters * filter_group_size for num_filters in filter_config],
        avg_seq_len,
        model_config.hidden_size,
        model_config.attention_head_size,
    )
    return mac


def main():
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_name
    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.ckpt_dir,
            "ltp",
            f"lambda_{args.reg_lambda}",
            f"temp_{args.init_temp}",
            f"lr_{args.lr}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

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
    
    config = MODELS[args.model_name][0].from_pretrained(args.ckpt_dir)
    config.prune_tokens = True
    model = MODELS[args.model_name][1].from_pretrained(args.ckpt_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        use_auth_token=None,
    )

    for param in model.parameters():
        param.requires_grad = False

    thresholds = []
    for name, param in model.named_parameters():
        if "threshold" in name:
            param.requires_grad = True
            thresholds.append(param)
    threshold_optimizer = AdamW(thresholds, lr=args.lr)

    mask_threshold = 0.5
    head_scores = torch.load(os.path.join(args.ckpt_dir, "head_scores.pt"))
    filter_scores = torch.load(os.path.join(args.ckpt_dir, "filter_scores.pt"))
    head_masks = [
        (score > mask_threshold).float().cuda() for score in head_scores
    ]
    num_heads = [mask.sum().item() for mask in head_masks]
    filter_masks = [
        (score > mask_threshold).float().cuda() for score in filter_scores
    ]
    num_filters = [mask.sum().item() for mask in filter_masks]

    train_dataloader = glue_dataloader(
        args.task_name,
        tokenizer,
        training=True,
        batch_size=args.batch_size,
    )
    eval_dataloader = glue_dataloader(
        args.task_name,
        tokenizer,
        training=False,
        batch_size=256,
    )

    total_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=threshold_optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps,
    )

    logger.info("***** Pruning *****")
    logger.info(f"  Num training examples = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Total batch size = {args.batch_size}")

    progress_bar = tqdm(
        range(total_training_steps),
    )
    metric = load_metric("glue", args.task_name)

    model = model.cuda()
    num_steps = 0
    for epoch in range(args.num_epochs):
        model.train()
        threshold_optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            for k, v in batch.items():
                batch[k] = v.to("cuda", non_blocking=True)
            batch_size = batch["attention_mask"].shape[0]
            avg_seq_len = (batch["attention_mask"].sum() / batch_size).item()
            baseline_mac = config_to_mac(config, num_heads, num_filters, avg_seq_len)
            temp = schedule_temp(
                num_steps,
                total_training_steps,
                init_temp=args.init_temp,
            )
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                head_masks=head_masks,
                filter_masks=filter_masks,
                labels=batch["labels"],
                temp=temp,
            )
            seq_len = [avg_seq_len] + [layer.token_pruner.num_tokens for layer in model.bert.encoder.layer]
            mac = config_to_mac(config, num_heads, num_filters, seq_len)
            reg_loss = mac / baseline_mac
            loss = logits.loss + args.reg_lambda * reg_loss
            loss.backward()

            threshold_optimizer.step()
            threshold_optimizer.zero_grad()
            lr_scheduler.step()
            progress_bar.update(1)
            num_steps += 1

        avg_mac = AverageMeter("MAC Ratio")
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to("cuda", non_blocking=True)
                batch_size = batch["attention_mask"].shape[0]
                avg_seq_len = (batch["attention_mask"].sum() / batch_size).item()
                baseline_mac = config_to_mac(config, [config.num_attention_heads] * config.num_hidden_layers, [config.num_filter_groups] * config.num_hidden_layers, avg_seq_len)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    head_masks=head_masks,
                    filter_masks=filter_masks,
                )
                seq_len = [avg_seq_len] + [layer.token_pruner.num_tokens for layer in model.bert.encoder.layer]
                mac = config_to_mac(config, num_heads, num_filters, seq_len)
                if model.problem_type == "regression":
                    predictions = outputs.logits.squeeze()
                else:
                    predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )
                avg_mac.update(mac / baseline_mac * 100.0, n=batch_size)
            curr_thresholds = [threshold.item() for threshold in thresholds]
            logger.info(f"Num heads: {num_heads}, Num filters: {num_filters}, Thresholds: {curr_thresholds}, Seq_len: {seq_len} {avg_mac} %")
            eval_metric = metric.compute()
            logger.info(f"epoch {epoch}: {eval_metric}")

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
