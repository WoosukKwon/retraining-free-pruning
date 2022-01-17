import torch

from dataset.glue import glue_dataloader
from dataset.squad import squad_test_dataloader
from evaluate.glue import eval_glue_acc
from evaluate.squad import eval_squad_acc


@torch.no_grad()
def test_accuracy(model, head_mask, neuron_mask, tokenizer, task_name):
    IS_SQUAD = "squad" in task_name

    test_batch_size = 32 if IS_SQUAD else 128
    if IS_SQUAD:
        eval_dataset, eval_examples, test_dataloader = squad_test_dataloader(
            task_name,
            tokenizer,
            batch_size=test_batch_size,
            pad_to_max=False,
        )
        acc = eval_squad_acc(
            model,
            head_mask,
            neuron_mask,
            test_dataloader,
            eval_dataset,
            eval_examples,
            task_name,
        )
    else:
        test_dataloader = glue_dataloader(
            task_name,
            tokenizer,
            training=False,
            batch_size=test_batch_size,
            pad_to_max=False,
        )
        acc = eval_glue_acc(
            model,
            head_mask,
            neuron_mask,
            test_dataloader,
            task_name,
        )
    return acc
