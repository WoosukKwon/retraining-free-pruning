import torch


@torch.no_grad()
def remove_padding(hidden_states, attention_mask):
    attention_mask = attention_mask.view(-1)
    nonzero = torch.nonzero(attention_mask).squeeze()
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[2])
    hidden_states = torch.index_select(hidden_states, dim=0, index=nonzero).contiguous()
    return hidden_states


def get_backbone(model):
    model_type = model.base_model_prefix
    backbone = getattr(model, model_type)
    return backbone


def get_encoder(model):
    backbone = get_backbone(model)
    encoder = backbone.encoder
    return encoder


def get_layers(model):
    encoder = get_encoder(model)
    layers = encoder.layer
    return layers


def get_mha_proj(model, index):
    layer = get_layers(model)[index]
    mha_proj = layer.attention.output
    return mha_proj


def get_ffn1(model, index):
    layer = get_layers(model)[index]
    ffn1 = layer.intermediate
    return ffn1


def get_ffn2(model, index):
    layer = get_layers(model)[index]
    ffn2 = layer.output
    return ffn2


def get_classifier(model):
    backbone = get_backbone(model)
    if backbone.pooler is not None:
        classifier = model.classifier
    else:
        classifier = model.classifier.out_proj
    return classifier


def register_mask(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask, inputs[1])
    handle = module.register_forward_pre_hook(hook)
    return handle


def apply_neuron_mask(model, neuron_mask):
    num_hidden_layers = neuron_mask.shape[0]
    handles = []
    for layer_idx in range(num_hidden_layers):
        ffn2 = get_ffn2(model, layer_idx)
        handle = register_mask(ffn2, neuron_mask[layer_idx])
        handles.append(handle)
    return handles


class MaskNeurons:
    def __init__(self, model, neuron_mask):
        self.handles = apply_neuron_mask(model, neuron_mask)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        for handle in self.handles:
            handle.remove()


def hijack_input(module, list_to_append):
    hook = lambda _, inputs: list_to_append.append(inputs)
    handle = module.register_forward_pre_hook(hook)
    return handle


@torch.no_grad()
def collect_layer_inputs(
    model,
    head_mask,
    neuron_mask,
    layer_idx,
    prev_inputs,
):
    layers = get_layers(model)
    target_layer = layers[layer_idx]

    inputs = []
    if layer_idx == 0:
        encoder = get_encoder(model)
        layers = encoder.layer
        encoder.layers = layers[:1]

        handle = hijack_input(target_layer, inputs)
        for batch in prev_inputs:
            for k, v in batch.items():
                batch[k] = v.to("cuda")
            with MaskNeurons(model, neuron_mask):
                model(head_mask=head_mask, **batch)

        handle.remove()
        encoder.layers = layers
        inputs = [list(x) for x in inputs]
    else:
        prev_layer = layers[layer_idx - 1]

        for batch in prev_inputs:
            batch[2] = head_mask[layer_idx - 1].view(1, -1, 1, 1)
            with MaskNeurons(model, neuron_mask):
                prev_output = prev_layer(*batch)

            batch[0] = prev_output[0]
            batch[2] = head_mask[layer_idx].view(1, -1, 1, 1)
            inputs.append(batch)

    return inputs
