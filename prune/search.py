import torch

from prune.fisher import compute_fisher_info
from efficiency.mac import compute_mac, mac_per_head, mac_per_neuron
from efficiency.latency import estimate_latency, fit_latency_fn


@torch.no_grad()
def search_mac(
    config,
    head_grads,
    neuron_grads,
    seq_len,
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
        num_neurons = max(num_neurons, 0)

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

    return head_mask, neuron_mask


@torch.no_grad()
def search_latency(
    config,
    head_grads,
    neuron_grads,
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
        torch.ones(num_hidden_layers, num_attention_heads),
        torch.ones(num_hidden_layers, intermediate_size),
    )
    max_latency = latency_constraint * original_latency

    mha_latency_fn = fit_latency_fn(mha_lut)
    ffn_latency_fn = fit_latency_fn(ffn_lut)

    head_importance = compute_fisher_info(head_grads)
    neuron_importance = compute_fisher_info(neuron_grads)

    # Locally rank heads and neurons
    local_head_importance, local_head_indicies = head_importance.sort(dim=1, descending=True)
    local_neuron_importance, local_neuron_indicies = neuron_importance.sort(dim=1, descending=True)

    base_head_indicies = local_head_indicies[:, :mha_latency_fn.threshold]
    base_neuron_indicies = local_neuron_indicies[:, :ffn_latency_fn.threshold]

    base_head_importance = local_head_importance[:, :mha_latency_fn.threshold].sum(dim=1)
    base_neuron_importance = local_neuron_importance[:, :ffn_latency_fn.threshold].sum(dim=1)

    _, mha_sorted_indicies = base_head_importance.sort(descending=False)
    _, ffn_sorted_indicies = base_neuron_importance.sort(descending=False)

    mha_offset = torch.arange(0, end=num_hidden_layers).unsqueeze(1).cuda() * num_attention_heads
    ffn_offset = torch.arange(0, end=num_hidden_layers).unsqueeze(1).cuda() * intermediate_size

    base_head_indicies = base_head_indicies + mha_offset
    base_neuron_indicies = base_neuron_indicies + ffn_offset

    orig_neuron_importance = neuron_importance.clone()
    max_importance = 0
    for num_mha_drops in range(num_hidden_layers):
        head_importance[mha_sorted_indicies[:num_mha_drops]] = 0
        remaining_base_head_indicies = base_head_indicies[mha_sorted_indicies[num_mha_drops:]].flatten()
        num_mha_layers = num_hidden_layers - num_mha_drops

        neuron_importance = orig_neuron_importance.clone()
        for num_ffn_drops in range(num_hidden_layers):
            neuron_importance[ffn_sorted_indicies[:num_ffn_drops]] = 0
            remaining_base_neuron_indicies = base_neuron_indicies[ffn_sorted_indicies[num_ffn_drops:]].flatten()
            num_ffn_layers = num_hidden_layers - num_ffn_drops

            remaining_head_indicies = local_head_indicies[:, mha_latency_fn.threshold:]
            remaining_neuron_indicies = local_neuron_indicies[:, ffn_latency_fn.threshold:]

            remaining_head_importance = head_importance.gather(dim=1, index=remaining_head_indicies)
            remaining_neuron_importance = neuron_importance.gather(dim=1, index=remaining_neuron_indicies)

            # Globally rank the remaining heads and neurons
            sorted_head_importance, sorted_head_indicies = remaining_head_importance.view(-1).sort(descending=True)
            sorted_neuron_importance, sorted_neuron_indicies = remaining_neuron_importance.view(-1).sort(descending=True)

            max_latency = max_latency - (num_mha_layers * mha_latency_fn.c + num_ffn_layers * ffn_latency_fn.c)
            if max_latency < 0:
                continue

            num_remaining_heads = num_mha_layers * (num_attention_heads - mha_latency_fn.threshold)
            for num_heads in range(1, num_remaining_heads + 1):
                heads_latency = mha_latency_fn.slope * num_heads
                neurons_latency = max_latency - heads_latency
                num_neurons = int(neurons_latency / ffn_latency_fn.slope)
                if num_neurons < 0:
                    break

                total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
                if total_importance > max_importance:
                    max_importance = total_importance

                    head_indicies = sorted_head_indicies[:num_heads]
                    head_indicies = (remaining_head_indicies + mha_offset).flatten()[head_indicies]
                    head_indicies = torch.cat([remaining_base_head_indicies, head_indicies], dim=0)

                    neuron_indicies = sorted_neuron_indicies[:num_neurons]
                    neuron_indicies = (remaining_neuron_indicies + ffn_offset).flatten()[neuron_indicies]
                    neuron_indicies = torch.cat([remaining_base_neuron_indicies, neuron_indicies], dim=0)

    head_mask = torch.zeros(num_hidden_layers, num_attention_heads).flatten()
    head_mask[head_indicies] = 1.0
    head_mask = head_mask.view(num_hidden_layers, num_attention_heads)

    neuron_mask = torch.zeros(num_hidden_layers, intermediate_size).flatten()
    neuron_mask[neuron_indicies] = 1.0
    neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)

    head_mask = head_mask.cuda()
    neuron_mask = neuron_mask.cuda()
    return head_mask, neuron_mask
