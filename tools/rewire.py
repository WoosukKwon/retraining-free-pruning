import torch


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


def importance_by_magnitude(model, config):
    partition_by_heads = lambda weight: weight.view(
        config.num_attention_heads,
        config.attention_head_size,
        config.hidden_size,
    ).reshape(config.num_attention_heads, -1)

    base_model = getattr(model, model.base_model_prefix)
    layers = base_model.encoder.layer

    head_importance = []
    for layer in layers:
        self_attention = layer.attention.self
        self_output = layer.attention.output

        query_per_head = partition_by_heads(self_attention.query.weight)
        key_per_head = partition_by_heads(self_attention.key.weight)
        value_per_head = partition_by_heads(self_attention.value.weight)
        out_per_head = partition_by_heads(self_output.dense.weight.t())

        abs_sum = torch.abs(query_per_head) + torch.abs(key_per_head) + torch.abs(value_per_head) + torch.abs(out_per_head)
        importance = abs_sum.sum(dim=1)
        head_importance.append(importance)

    neuron_importance = []
    for layer in layers:
        ffn1_per_neuron = layer.intermediate.dense.weight
        ffn2_per_neuron = layer.output.dense.weight.t()
        abs_sum = torch.abs(ffn1_per_neuron) + torch.abs(ffn2_per_neuron)
        importance = abs_sum.sum(dim=1)
        neuron_importance.append(importance)

    return head_importance, neuron_importance


def rewire_by_magnitude(model):
    base_model = getattr(model, model.base_model_prefix)
    config = base_model.config
    head_importance, neuron_importance = importance_by_magnitude(model, config)
    rewire(base_model, head_importance, neuron_importance)
