import torch

from tools.importance import importance_by_magnitude, importance_by_gradient


def reorder_linear(linear, indicies, dim):
    indicies = indicies.to(linear.weight.device)
    reordered_weight = linear.weight.index_select(dim, indicies)
    linear.weight.copy_(reordered_weight.contiguous())

    if linear.bias is not None and dim == 0:
        reordered_bias = linear.bias[indicies]
        linear.bias.copy_(reordered_bias.contiguous())


def reorder_heads(layer_module, indicies):
    self_attention = layer_module.attention.self
    indicies = torch.arange(
        self_attention.all_head_size
    ).view(
        self_attention.num_attention_heads,
        self_attention.attention_head_size
    )[indicies]
    indicies = torch.flatten(indicies).long()

    reorder_linear(self_attention.query, indicies, dim=0)
    reorder_linear(self_attention.key, indicies, dim=0)
    reorder_linear(self_attention.value, indicies, dim=0)

    attention_output = layer_module.attention.output
    reorder_linear(attention_output.dense, indicies, dim=1)


def reorder_neurons(layer_module, indicies):
    reorder_linear(layer_module.intermediate.dense, indicies, dim=0)
    reorder_linear(layer_module.output.dense, indicies, dim=1)


@torch.no_grad()
def rewire(base_model, head_importance, neuron_importance):
    for i in range(base_model.config.num_hidden_layers):
        indicies = torch.argsort(head_importance[i], descending=True)
        reorder_heads(base_model.encoder.layer[i], indicies)

        indicies = torch.argsort(neuron_importance[i], descending=True)
        reorder_neurons(base_model.encoder.layer[i], indicies)


def rewire_by_magnitude(model):
    base_model = getattr(model, model.base_model_prefix)
    config = base_model.config
    head_importance, neuron_importance = importance_by_magnitude(model, config)
    rewire(base_model, head_importance, neuron_importance)


def rewire_by_gradient(model, dataloader, absolute=True):
    base_model = getattr(model, model.base_model_prefix)
    config = base_model.config
    head_importance, neuron_importance = importance_by_gradient(model, config, dataloader, absolute=absolute)
    rewire(base_model, head_importance, neuron_importance)
