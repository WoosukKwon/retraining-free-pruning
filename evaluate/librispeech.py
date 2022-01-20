from tqdm import tqdm
import numpy as np
from datasets import load_metric

import torch
from torch.utils.data import DataLoader
from utils.arch import apply_neuron_mask
from dataset.glue import target_dev_metric


@torch.no_grad()
def test_wer(model, head_mask, neuron_mask, dataset, collate_fn, processor):
    test_batch_size = 16
    test_dataloader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    wer = eval_librispeech_wer(
        model,
        head_mask,
        neuron_mask,
        test_dataloader,
        processor,
    )
    return wer

@torch.no_grad()
def eval_librispeech_wer(model, head_mask, neuron_mask, dataloader, processor):
    metric = load_metric("wer")

    model.eval()
    #handles = apply_neuron_mask(model, neuron_mask)
    pred_strs, label_strs = [], []
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        pred = model(**batch, head_mask=head_mask)
        pred_logits = pred.logits.cpu()

        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        
        label_ids = batch['labels']
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        pred_strs += pred_str
        label_strs += label_str

    for p, l in zip(pred_strs[:50], label_strs[:50]):
        print(p)
        print(l)
    wer = metric.compute(predictions=pred_strs, references=label_strs)

    return wer
