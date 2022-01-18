import torch
from datasets import load_metric

from dataset.squad import create_and_fill_np_array, post_processing_function
from utils.arch import apply_neuron_mask
from utils.meter import AverageMeter


@torch.no_grad()
def eval_squad_acc(
    model,
    head_mask,
    neuron_mask,
    dataloader,
    eval_dataset,
    eval_examples,
    task_name,
):
    metric = load_metric(task_name)

    model.eval()
    handles = apply_neuron_mask(model, neuron_mask)
    all_start_logits = []
    all_end_logits = []
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        outputs = model(head_mask=head_mask, **batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        all_start_logits.append(start_logits.cpu().numpy())
        all_end_logits.append(end_logits.cpu().numpy())
    for handle in handles:
        handle.remove()

    max_len = max([x.shape[1] for x in all_start_logits])
    start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(task_name, eval_examples, eval_dataset, outputs_numpy)
    eval_results = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    accuracy = eval_results["f1"]
    return accuracy


@torch.no_grad()
def eval_squad_loss(
    model,
    head_mask,
    neuron_mask,
    dataloader,
):
    loss = AverageMeter("squad_loss")

    model.eval()
    handles = apply_neuron_mask(model, neuron_mask)
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        outputs = model(head_mask=head_mask, **batch)
        loss.update(outputs.loss, n=batch["input_ids"].shape[0])
    for handle in handles:
        handle.remove()

    return loss.avg
