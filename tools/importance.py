import torch
from tqdm import tqdm


def importance_by_gradient(model, config, dataloader, postprocess=lambda x: x * x):
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_filter_groups = config.num_filter_groups

    head_mask = torch.ones(num_hidden_layers, num_attention_heads).cuda()
    head_mask.requires_grad_(True)
    head_importance = torch.zeros(num_hidden_layers, num_attention_heads).cuda()

    filter_mask = torch.ones(num_hidden_layers, num_filter_groups).cuda()
    filter_mask.requires_grad_(True)
    filter_importance = torch.zeros(num_hidden_layers, num_filter_groups).cuda()

    model.eval()
    for batch in tqdm(dataloader):
        batch["head_masks"] = head_mask
        batch["filter_masks"] = filter_mask
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        head_grad = head_mask.grad.detach()
        head_mask.grad = None
        head_importance += postprocess(head_grad)

        filter_grad = filter_mask.grad.detach()
        filter_mask.grad = None
        filter_importance += postprocess(filter_grad)

    return head_importance, filter_importance


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
