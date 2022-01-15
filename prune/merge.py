import torch

from utils.arch import (
    get_ffn2,
    get_layers,
    hijack_input,
    remove_padding,
    collect_layer_inputs,
)


@torch.no_grad()
def merge_neurons(
    model,
    head_mask,
    neuron_mask,
    threshold,
    dataloader,
):
    num_hidden_layers = neuron_mask.shape[0]
    intermediate_size = neuron_mask.shape[1]

    for layer_idx in range(num_hidden_layers):
        layer_inputs = collect_layer_inputs(
            model,
            head_mask,
            neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else layer_inputs,
        )

        ffn2 = get_ffn2(model, layer_idx)
        ffn2_inputs = []
        handle = hijack_input(ffn2, ffn2_inputs)

        num_tokens = 0
        pdist = torch.nn.PairwiseDistance(p=2, keepdim=False)
        l2_dist = torch.zeros(intermediate_size, intermediate_size).cuda()
        layer = get_layers(model)[layer_idx]
        for batch in layer_inputs:
            attention_mask = (batch[1] == 0)
            layer(*batch) # NOTE: do not apply neuron_mask

            hidden_states, _ = ffn2_inputs.pop(0)
            hidden_states = remove_padding(hidden_states, attention_mask)
            hidden_states = hidden_states.t()

            num_tokens += hidden_states.shape[1]

            # FIXME: performance
            for i in range(intermediate_size):
                l2_dist[i] += pdist(hidden_states, hidden_states[i])

        handle.remove()
        l2_dist = l2_dist / num_tokens

        nonzero_indicies = torch.nonzero(neuron_mask[layer_idx]).flatten()
        zero_indicies = torch.nonzero(neuron_mask[layer_idx] == 0).flatten()

        if len(nonzero_indicies) == 0 or len(zero_indicies) == 0:
            continue

        for pruned_idx in zero_indicies:
            neuron_dist = l2_dist[pruned_idx][nonzero_indicies]
            if neuron_dist.min() > threshold:
                continue

            closest = nonzero_indicies[neuron_dist.argmin()]
            ffn2.dense.weight[:, closest] += ffn2.dense.weight[:, pruned_idx]
