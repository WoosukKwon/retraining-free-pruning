import argparse
import copy
import logging
import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoTokenizer, default_data_collator, set_seed

from models.bert.config import BertConfig
from models.bert.model import BertForSequenceClassification
from tools.glue import glue_dataloader, max_seq_length, glue_dataset
from tools.mac import compute_mac


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
parser.add_argument("--search_algo", required=True, choices=[
    "greedy",
    "beam",
])
parser.add_argument("--max_acc_drop", type=float, default=1.0)
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


@torch.no_grad()
def test(model, head_config, filter_config, test_dataloader, metric):
    head_masks = config_to_masks(head_config, 12)
    filter_masks = config_to_masks(filter_config, 12)

    for batch in tqdm(test_dataloader):
        outputs = model(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
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
    logger.info(eval_metric)


@torch.no_grad()
def config_to_masks(mask_config, max_num):
    seq = torch.arange(1, max_num + 1)
    masks = [(seq <= num).float().cuda() for num in mask_config]
    return masks


def config_to_mac(model_config, head_config, filter_config, avg_seq_len):
    filter_group_size = int(model_config.intermediate_size / model_config.num_attention_heads) # Note: num_groups == num_attention_heads
    mac = compute_mac(
        head_config,
        [num_filters * filter_group_size for num_filters in filter_config],
        avg_seq_len,
        model_config.hidden_size,
        model_config.attention_head_size,
    )
    return mac


def greedy_search(model, model_config, sample_batch, metric, avg_seq_len, mac_acc_drop):
    num_hidden_layers = model_config.num_hidden_layers
    num_attention_heads = model_config.num_attention_heads
    num_filter_groups = model_config.num_attention_heads

    head_config = [num_attention_heads] * num_hidden_layers
    filter_config = [num_filter_groups] * num_hidden_layers

    head_masks = config_to_masks(head_config, num_attention_heads)
    filter_masks = config_to_masks(filter_config, num_filter_groups)

    base_acc = evaluate(model, sample_batch, head_masks, filter_masks, metric)
    acc_threshold = base_acc - mac_acc_drop
    base_mac = config_to_mac(model_config, head_config, filter_config, avg_seq_len)
    logger.info(f"Base accuracy: {base_acc:.2f}, Accuracy threshold: {acc_threshold:.2f}")

    curr_acc = base_acc
    for i in range(2 * num_hidden_layers):
        # Start from the last layer
        layer_idx = 11 - int(i / 2)
        if i % 2 == 0:
            # FFN
            head_masks = config_to_masks(head_config, num_attention_heads)
            for num_groups in range(num_filter_groups):
                tmp_filter_config = copy.deepcopy(filter_config)
                tmp_filter_config[layer_idx] = num_groups
                tmp_filter_masks = config_to_masks(tmp_filter_config, num_filter_groups)

                acc = evaluate(model, sample_batch, head_masks, tmp_filter_masks, metric)
                if acc >= acc_threshold:
                    filter_config[layer_idx] = num_groups
                    curr_acc = acc
                    break
            logger.info(f"Iteration {i}: Layer {layer_idx} - NUM FILTER GROUPS: {filter_config[layer_idx]} Acc: {curr_acc:.2f}")
        else:
            # MHA
            filter_masks = config_to_masks(filter_config, num_filter_groups)
            for num_heads in range(num_attention_heads):
                tmp_head_config = copy.deepcopy(head_config)
                tmp_head_config[layer_idx] = num_heads
                tmp_head_masks = config_to_masks(tmp_head_config, num_attention_heads)

                acc = evaluate(model, sample_batch, tmp_head_masks, filter_masks, metric)
                if acc >= acc_threshold:
                    head_config[layer_idx] = num_heads
                    curr_acc = acc
                    break
            logger.info(f"Iteration {i}: Layer {layer_idx} - NUM HEADS: {head_config[layer_idx]} Acc: {curr_acc:.2f}")

    mac = config_to_mac(model_config, head_config, filter_config, avg_seq_len)
    reduced_ratio = mac / base_mac
    logger.info(f"Original MAC: {base_mac / 1000000:.2f} M, Reduced MAC: {mac / 1000000:.2f} ({reduced_ratio * 100.0:.2f} %)")

    return head_config, filter_config


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
            f"acc_drop_{args.max_acc_drop}",
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
    attention_mask = sample_batch["attention_mask"]
    avg_seq_len = attention_mask.sum(dim=1).float().mean().item()
    metric = load_metric("glue", args.task_name)

    model = model.cuda()
    sample_batch["input_ids"] = sample_batch["input_ids"].cuda()
    sample_batch["attention_mask"] = sample_batch["attention_mask"].cuda()
    sample_batch["labels"] = sample_batch["labels"].cuda()

    if args.search_algo == "greedy":
        head_config, filter_config = greedy_search(
            model,
            config,
            sample_batch,
            metric,
            avg_seq_len,
            mac_acc_drop=args.max_acc_drop,
        )

    test_dataloader = glue_dataloader(
        args.task_name,
        tokenizer=tokenizer,
        training=False,
        batch_size=128,
    )
    test(model, head_config, filter_config, test_dataloader, metric)


if __name__ == "__main__":
    main()
