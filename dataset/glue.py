import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import default_data_collator
from transformers.data.data_collator import DataCollatorWithPadding


GLUE_TASKS = [
    "stsb",
    "mrpc",
    "rte",
    "sst2",
    "qqp",
    "qnli",
    "cola",
    "mnli",
    "mnli-m",
    "mnli-mm",
]


def num_labels(task_name):
    TASK_TO_NUM_LABELS = {
        "stsb": 1,
        "mrpc": 2,
        "rte": 2,
        "sst2": 2,
        "qqp": 2,
        "qnli": 2,
        "cola": 2,
        "mnli": 3,
        "mnli-m": 3,
        "mnli-mm": 3,
    }
    return TASK_TO_NUM_LABELS[task_name]


def max_seq_length(task_name):
    TASK_TO_SEQ_LEN = {
        "stsb": 128,
        "mrpc": 128,
        "rte": 128,
        "sst2": 64,
        "qqp": 128,
        "qnli": 128,
        "cola": 64,
        "mnli": 128,
        "mnli-m": 128,
        "mnli-mm": 128,
    }
    return TASK_TO_SEQ_LEN[task_name]


def avg_seq_length(task_name):
    # Dev set
    TASK_TO_SEQ_LEN = {
        "stsb": 31.47,
        "mrpc": 53.24,
        "rte": 64.59,
        "sst2": 25.16,
        "qqp": 30.55,
        "qnli": 50.97,
        "cola": 11.67,
        "mnli": 39.05,
    }
    return TASK_TO_SEQ_LEN[task_name]


def target_dev_metric(task_name):
    TASK_TO_DEV_METRIC = {
        "stsb": "spearmanr",
        "mrpc": "accuracy",
        "rte": "accuracy",
        "sst2": "accuracy",
        "qqp": "accuracy",
        "qnli": "accuracy",
        "cola": "matthews_correlation",
        "mnli": "accuracy",
        "mnli-m": "accuracy",
        "mnli-mm": "accuracy",
    }
    return TASK_TO_DEV_METRIC[task_name]


def preprocess_glue(examples, tokenizer, sentence_keys, max_seq_len, pad_to_max, label_key="label"):
    sentence1_key, sentence2_key = sentence_keys
    if sentence2_key is None:
        args = (examples[sentence1_key], None)
    else:
        args = (examples[sentence1_key], examples[sentence2_key])
    result = tokenizer(
        *args,
        padding="max_length" if pad_to_max else False,
        max_length=max_seq_len,
        truncation=True,
    )
    if isinstance(examples, dict):
        if label_key in examples.keys():
            result["labels"] = examples[label_key]
    else:
        result["labels"] = examples[label_key]
    return result


def glue_dataset(task_name, tokenizer, training, max_seq_len, pad_to_max=False):
    TASK_TO_KEYS = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    keys = TASK_TO_KEYS[task_name]

    raw_datasets = load_dataset("glue", task_name)
    preprocessed = raw_datasets.map(
        lambda examples: preprocess_glue(examples, tokenizer, keys, max_seq_len, pad_to_max),
        batched=True,
        load_from_cache_file=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    if training:
        return preprocessed["train"]
    else:
        return preprocessed["validation_matched" if task_name == "mnli" else "validation"]


def glue_dataloader(task_name, tokenizer, training, batch_size=32, max_seq_len=None, pad_to_max=False):
    if max_seq_len is None:
        max_seq_len = max_seq_length(task_name)
    dataset = glue_dataset(task_name, tokenizer, training, max_seq_len, pad_to_max=pad_to_max)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        collate_fn=default_data_collator if pad_to_max else DataCollatorWithPadding(tokenizer),
    )
    return dataloader
