import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction,
)

from tools.qa_utils import postprocess_qa_predictions


def prepare_validation_features(
        examples,
        tokenizer,
        question_column_name,
        context_column_name,
        max_seq_len,
        pad_to_max=False,
        doc_stride=128,
    ):
    pad_on_right = tokenizer.padding_side == "right"

    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max else False,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def squad_dataset(task_name, tokenizer, training, max_seq_len, pad_to_max=False):
    assert not training, "Currently only support inference"
    raw_datasets = load_dataset(task_name)
    column_names = raw_datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]

    eval_examples = raw_datasets["validation"]
    eval_dataset = eval_examples.map(
        lambda examples: prepare_validation_features( # FIXME
            examples,
            tokenizer,
            question_column_name,
            context_column_name,
            max_seq_len,
            pad_to_max,
        ),
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
    )
    return eval_dataset, eval_examples


def squad_dataloader(task_name, tokenizer, training, batch_size, max_seq_len=384, pad_to_max=False):
    dataset, _ = squad_dataset(task_name, tokenizer, training, max_seq_len, pad_to_max=pad_to_max)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        collate_fn=default_data_collator if pad_to_max else DataCollatorWithPadding(tokenizer),
    )
    return dataloader


def post_processing_function(task_name, examples, dataset, predictions, stage="eval"):
    answer_column_name = "answers"
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=dataset,
        predictions=predictions,
        version_2_with_negative=task_name == "squad_v2",
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        prefix=stage,
    )
    if task_name == "squad_v2":
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)
