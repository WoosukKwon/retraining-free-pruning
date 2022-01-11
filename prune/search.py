import torch

from prune.fisher import collect_mask_grads, compute_fisher_info
from efficiency.mac import compute_mac, mac_per_head, mac_per_neuron
from prune.rearrange import greedy_rearrange


def search_mac(model, config, seq_len, dataloader, mac_constraint):
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

    ones_head_mask = torch.ones(num_hidden_layers, num_attention_heads).cuda()
    ones_neuron_mask = torch.ones(num_hidden_layers, intermediate_size).cuda()

    head_grads, neuron_grads = collect_mask_grads(model, ones_head_mask, ones_neuron_mask, dataloader)
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
    # NOTE: temporarily convert to CPU tensors as the arithmetic intensity is very low
    head_mask = head_mask.cpu()
    neuron_mask = neuron_mask.cpu()
    head_grads = head_grads.cpu()
    neuron_grads = neuron_grads.cpu()

    for i in range(num_hidden_layers):
        head_mask[i] = greedy_rearrange(head_mask[i], head_grads[:, i, :])
        neuron_mask[i] = greedy_rearrange(neuron_mask[i], neuron_grads[:, i, :])

    head_mask = head_mask.cuda()
    neuron_mask = neuron_mask.cuda()
    return head_mask, neuron_mask
