import argparse
import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)

from models.bert.config import BertConfig
from models.bert.model import BertForQuestionAnswering
from tools.squad import squad_dataset, post_processing_function
from tools.qa_utils import create_and_fill_np_array


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
    metric = load_metric(args.task_name)

    eval_dataset, eval_examples = squad_dataset(args.task_name, tokenizer, training=False, max_seq_len=384, pad_to_max=False)
    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model,
        batch_size=128,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    config = MODELS[args.model_name][0].from_pretrained(args.ckpt_dir)
    model = MODELS[args.model_name][1].from_pretrained(args.ckpt_dir, config=config)

    accelerator = Accelerator()
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    all_start_logits = []
    all_end_logits = []

    model.eval()
    with torch.no_grad():
        head_masks = [torch.ones(config.num_attention_heads).cuda() for _ in range(config.num_hidden_layers)]
        filter_masks = [torch.ones(config.num_filter_groups).cuda() for _ in range(config.num_hidden_layers)]

        for batch in tqdm(eval_dataloader):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                head_masks=head_masks,
                filter_masks=filter_masks,
            )
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())
        
        max_len = max([x.shape[1] for x in all_start_logits])
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(args.task_name, eval_examples, eval_dataset, outputs_numpy)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)

    print(eval_metric)


if __name__ == "__main__":
    main()
