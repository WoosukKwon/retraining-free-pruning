import argparse
import logging
import os

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Function
from datasets import load_metric
from transformers import AutoTokenizer, set_seed
from transformers import AdamW, get_scheduler
from transformers.data.data_collator import DataCollatorWithPadding

from models.bert.config import BertConfig
from models.bert.model import BertForQuestionAnswering
from tools.mac import compute_mac
from tools.squad import squad_dataset
from tools.squad import squad_dataset, post_processing_function
from tools.qa_utils import create_and_fill_np_array


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
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--reg_lambda", type=float, default=0.5)
parser.add_argument("--init_threshold", type=float, default=0.8)
parser.add_argument("--final_threshold", type=float, default=0.5)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default=None)


def schedule_threshold(
    step: int,
    total_step: int,
    initial_threshold: float,
    final_threshold: float,
    final_lambda: float,
):
    mul_coeff = 1 - step / total_step
    threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
    reg_lambda = final_lambda * threshold / final_threshold
    return threshold, reg_lambda


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


class STEMask(Function):

    @staticmethod
    def forward(ctx, score, threshold):
        return (score > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def main():
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_name
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "logs",
            "mvp",
            args.model_name,
            args.task_name,
            f"lambda_{args.reg_lambda}",
            f"init_threshold_{args.init_threshold}",
            f"batch_{args.batch_size}",
            f"lr_{args.lr}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.task_name == "squad":
        SEQ_LEN = 175.96
    elif args.task_name == "squad_v2":
        SEQ_LEN = 179.99

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
    model = MODELS[args.model_name][1].from_pretrained(args.ckpt_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        use_auth_token=None,
    )

    for param in model.parameters():
        param.requires_grad = False

    head_scores = [
        torch.ones(config.num_attention_heads, requires_grad=True, device="cuda") for _ in range(config.num_hidden_layers)
    ]
    filter_scores = [
        torch.ones(config.num_filter_groups, requires_grad=True, device="cuda") for _ in range(config.num_hidden_layers)
    ]
    score_optimizer = AdamW(head_scores + filter_scores, lr=args.lr)

    train_dataset = squad_dataset(args.task_name, tokenizer, training=True, max_seq_len=384, pad_to_max=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    eval_dataset, eval_examples = squad_dataset(args.task_name, tokenizer, training=False, max_seq_len=384, pad_to_max=False)
    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model,
        batch_size=128,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )    

    total_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=score_optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps,
    )

    logger.info("***** Movement Pruning *****")
    logger.info(f"  Num training examples = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Total batch size = {args.batch_size}")

    progress_bar = tqdm(
        range(total_training_steps),
    )
    metric = load_metric(args.task_name)
    baseline_mac = config_to_mac(
        config,
        [config.num_attention_heads] * config.num_hidden_layers,
        [config.num_filter_groups] * config.num_hidden_layers,
        SEQ_LEN,
    )

    model = model.cuda()
    num_steps = 0
    for epoch in range(args.num_epochs):
        model.train()
        score_optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            for k, v in batch.items():
                batch[k] = v.to("cuda", non_blocking=True)

            threshold, reg_lambda = schedule_threshold(
                num_steps,
                total_training_steps,
                initial_threshold=args.init_threshold,
                final_threshold=args.final_threshold,
                final_lambda=args.reg_lambda,
            )
            head_masks = [
                STEMask.apply(score, threshold) for score in head_scores
            ]
            filter_masks = [
                STEMask.apply(score, threshold) for score in filter_scores
            ]

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                head_masks=head_masks,
                filter_masks=filter_masks,
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"],
            )
            mac = config_to_mac(
                config,
                [mask.sum() for mask in head_masks],
                [mask.sum() for mask in filter_masks],
                SEQ_LEN,
            )
            reg_loss = mac / baseline_mac
            loss = logits.loss + reg_lambda * reg_loss
            loss.backward()

            score_optimizer.step()
            score_optimizer.zero_grad()
            lr_scheduler.step()
            progress_bar.update(1)
            num_steps += 1

        all_start_logits = []
        all_end_logits = []
        model.eval()
        with torch.no_grad():
            head_masks = [
                STEMask.apply(score, threshold) for score in head_scores
            ]
            filter_masks = [
                STEMask.apply(score, threshold) for score in filter_scores
            ]
            num_heads = [mask.sum().item() for mask in head_masks]
            num_filters = [mask.sum().item() for mask in filter_masks]
            mac = config_to_mac(config, num_heads, num_filters, SEQ_LEN)
            logger.info(f"Num heads: {num_heads}, Num filters: {num_filters}, MAC: {mac / baseline_mac * 100:.2f} %")

            for step, batch in enumerate(eval_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to("cuda", non_blocking=True)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    head_masks=head_masks,
                    filter_masks=filter_masks,
                )
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                all_start_logits.append(start_logits.cpu().numpy())
                all_end_logits.append(end_logits.cpu().numpy())

            max_len = max([x.shape[1] for x in all_start_logits])
            start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
            end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)
            del all_start_logits
            del all_end_logits

            outputs_numpy = (start_logits_concat, end_logits_concat)
            prediction = post_processing_function(args.task_name, eval_examples, eval_dataset, outputs_numpy)
            eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
            logger.info(f"epoch {epoch}: {eval_metric}")

    model.save_pretrained(args.output_dir)
    torch.save(head_masks, os.path.join(args.output_dir, "head_masks.pt"))
    torch.save(filter_masks, os.path.join(args.output_dir, "filter_masks.pt"))


if __name__ == "__main__":
    main()
