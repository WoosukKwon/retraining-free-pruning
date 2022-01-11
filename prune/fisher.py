import torch

from utils.arch import apply_neuron_mask


def collect_mask_grads(model, head_mask, neuron_mask, dataloader):
    head_mask.requires_grad_(True)
    neuron_mask.requires_grad_(True)

    handles = apply_neuron_mask(model, neuron_mask)

    model.eval()
    head_grads = []
    neuron_grads = []
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        outputs = model(head_mask=head_mask, **batch)
        loss = outputs.loss
        loss.backward()

        head_grads.append(head_mask.grad.detach())
        head_mask.grad = None

        neuron_grads.append(neuron_mask.grad.detach())
        neuron_mask.grad = None

    for handle in handles:
        handle.remove()
    head_mask.requires_grad_(False)
    neuron_mask.requires_grad_(False)

    head_grads = torch.stack(head_grads, dim=0)
    neuron_grads = torch.stack(neuron_grads, dim=0)
    return head_grads, neuron_grads


@torch.no_grad()
def compute_fisher_info(grads):
    fisher_info = grads.pow(2).sum(dim=0)
    return fisher_info
