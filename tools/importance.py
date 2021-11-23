import torch
from tqdm import tqdm


def importance_by_gradient(model, config, dataloader, absolute=True):
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_filter_groups = config.num_filter_groups

    head_mask = torch.ones(num_hidden_layers, num_attention_heads).cuda() # FIXME
    head_mask.requires_grad_(True)
    head_importance = torch.zeros(num_hidden_layers, num_attention_heads).cuda() # FIXME

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []
    for name, w in model.named_parameters():
        if "intermediate" in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if "output" in name and "attention" not in name:
            if w.dim() > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).to(w.device))

    filter_mask = torch.ones(num_hidden_layers, num_filter_groups).cuda()
    model.eval()
    for batch in tqdm(dataloader):
        batch["head_masks"] = head_mask
        batch["filter_masks"] = filter_mask
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        head_importance += head_mask.grad.abs().detach() if absolute else head_mask.grad.detach()
        head_mask.grad = None
        for w1, b1, w2, importance in zip(
            intermediate_weight,
            intermediate_bias,
            output_weight,
            neuron_importance
        ):
            w1_importance = ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).detach()
            w2_importance = ((w2 * w2.grad).sum(dim=0)).detach()
            if absolute:
                w1_importance = w1_importance.abs()
                w2_importance = w2_importance.abs()
            importance += w1_importance + w2_importance
            w1.grad = None
            b1.grad = None
            w2.grad = None

    return head_importance, neuron_importance


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
