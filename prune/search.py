import torch

from prune.fisher import collect_mask_grads, compute_fisher_info
from prune.rearrange import greedy_rearrange
from efficiency.mac import compute_mac, mac_per_head, mac_per_neuron
from efficiency.latency import estimate_latency, fit_latency_fn


def rearrange_mask(mask, grads):
    # NOTE: temporarily convert to CPU tensors as the arithmetic intensity is very low
    device = mask.device
    mask = mask.cpu()
    grads = grads.cpu()

    num_hidden_layers = mask.shape[0]
    for i in range(num_hidden_layers):
        mask[i] = greedy_rearrange(mask[i], grads[:, i, :])

    mask = mask.to(device)
    return mask


def search_mac(
    model,
    config,
    prev_head_mask,
    prev_neuron_mask,
    seq_len,
    dataloader,
    mac_constraint,
):
    assert mac_constraint < 1

    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )
    max_mac = mac_constraint * original_mac

    head_grads, neuron_grads = collect_mask_grads(model, prev_head_mask, prev_neuron_mask, dataloader)
    head_importance = compute_fisher_info(head_grads)
    neuron_importance = compute_fisher_info(neuron_grads)

    # Globally rank heads and neurons
    sorted_head_importance, sorted_head_indicies = head_importance.view(-1).sort(descending=True)
    sorted_neuron_importance, sorted_neuron_indicies = neuron_importance.view(-1).sort(descending=True)

    max_importance = 0
    for num_heads in range(1, num_hidden_layers * num_attention_heads + 1):
        heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * num_heads
        neurons_mac = max_mac - heads_mac
        num_neurons = int(neurons_mac / mac_per_neuron(seq_len, hidden_size))

        total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
        if total_importance > max_importance:
            max_importance = total_importance
            head_indicies = sorted_head_indicies[:num_heads]
            neuron_indicies = sorted_neuron_indicies[:num_neurons]

    head_mask = torch.zeros(num_hidden_layers * num_attention_heads).cuda()
    head_mask[head_indicies] = 1.0
    head_mask = head_mask.view(num_hidden_layers, num_attention_heads)

    neuron_mask = torch.zeros(num_hidden_layers * intermediate_size).cuda()
    neuron_mask[neuron_indicies] = 1.0
    neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)

    # Rearrange the mask
    head_mask = rearrange_mask(head_mask, head_grads)
    neuron_mask = rearrange_mask(neuron_mask, neuron_grads)
    return head_mask, neuron_mask


def search_latency(
    model,
    config,
    prev_head_mask,
    prev_neuron_mask,
    dataloader,
    latency_constraint,
    mha_lut,
    ffn_lut,
):
    assert latency_constraint < 1

    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    original_latency = estimate_latency(
        mha_lut,
        ffn_lut,
        torch.ones_like(prev_head_mask),
        torch.ones_like(prev_neuron_mask),
    )
    max_latency = latency_constraint * original_latency

    mha_latency_fn = fit_latency_fn(mha_lut)
    ffn_latency_fn = fit_latency_fn(ffn_lut)

    head_grads, neuron_grads = collect_mask_grads(model, prev_head_mask, prev_neuron_mask, dataloader)
    head_importance = compute_fisher_info(head_grads)
    neuron_importance = compute_fisher_info(neuron_grads)

    # Must include at least one head/neuron per layer
    max_latency = max_latency - num_hidden_layers * mha_latency_fn.c
    max_latency = max_latency - num_hidden_layers * ffn_latency_fn.c
    assert max_latency > 0

    # Locally rank heads and neurons
    _, local_head_indicies = head_importance.sort(dim=1, descending=True)
    _, local_neuron_indicies = neuron_importance.sort(dim=1, descending=True)
    base_head_indicies = local_head_indicies[:, :mha_latency_fn.threshold]
    base_neuron_indicies = local_neuron_indicies[:, :ffn_latency_fn.threshold]

    remaining_head_indicies = local_head_indicies[:, mha_latency_fn.threshold:]
    remaining_neuron_indicies = local_neuron_indicies[:, ffn_latency_fn.threshold:]

    remaining_head_importance = head_importance.gather(dim=1, index=remaining_head_indicies)
    remaining_neuron_importance = neuron_importance.gather(dim=1, index=remaining_neuron_indicies)

    # Globally rank the remaining heads and neurons
    sorted_head_importance, sorted_head_indicies = remaining_head_importance.view(-1).sort(descending=True)
    sorted_neuron_importance, sorted_neuron_indicies = remaining_neuron_importance.view(-1).sort(descending=True)

    max_importance = 0
    num_remaining_heads = num_hidden_layers * (num_attention_heads - mha_latency_fn.threshold)
    for num_heads in range(1, num_remaining_heads + 1):
        heads_latency = mha_latency_fn.slope * num_heads
        neurons_latency = max_latency - heads_latency
        num_neurons = int(neurons_latency / ffn_latency_fn.slope)
        if num_neurons < 0:
            continue

        total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
        if total_importance > max_importance:
            max_importance = total_importance
            head_indicies = sorted_head_indicies[:num_heads]
            neuron_indicies = sorted_neuron_indicies[:num_neurons]

    offset = torch.arange(0, end=num_hidden_layers).unsqueeze(1).cuda() * num_attention_heads
    base_head_indicies += offset
    base_head_indicies = base_head_indicies.flatten()
    remaining_head_indicies += offset
    head_indicies = remaining_head_indicies.flatten()[head_indicies]
    head_indicies = torch.cat([base_head_indicies, head_indicies], dim=0)

    offset = torch.arange(0, end=num_hidden_layers).unsqueeze(1).cuda() * intermediate_size
    base_neuron_indicies += offset
    base_neuron_indicies = base_neuron_indicies.flatten()
    remaining_neuron_indicies += offset
    neuron_indicies = remaining_neuron_indicies.flatten()[neuron_indicies]
    neuron_indicies = torch.cat([base_neuron_indicies, neuron_indicies], dim=0)

    head_mask = torch.zeros_like(prev_head_mask).flatten()
    head_mask[head_indicies] = 1.0
    head_mask = head_mask.view(num_hidden_layers, num_attention_heads)

    neuron_mask = torch.zeros_like(prev_neuron_mask).flatten()
    neuron_mask[neuron_indicies] = 1.0
    neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)

    # Rearrange the mask
    head_mask = rearrange_mask(head_mask, head_grads)
    neuron_mask = rearrange_mask(neuron_mask, neuron_grads)
    return head_mask, neuron_mask
