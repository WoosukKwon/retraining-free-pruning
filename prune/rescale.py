from tqdm import tqdm
import torch

from utils.linalg import closed_form_solver, gmres_cupy_solver


from utils.arch import (
    get_encoder,
    get_layers,
    get_mha_proj,
    get_ffn2,
    hijack_input,
    MaskNeurons,
    remove_padding,
)


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


@torch.no_grad()
def get_mha_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
):
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    layer = get_layers(model)[layer_idx]
    mha_proj = get_mha_proj(model, layer_idx)
    weights_per_head = mha_proj.dense.weight.t().view(num_attention_heads, -1, hidden_size)

    inputs = []
    handle = hijack_input(mha_proj, inputs)

    ATA = torch.zeros(num_attention_heads, num_attention_heads).cuda()
    ATB = torch.zeros(num_attention_heads).cuda()

    model.eval()
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
        attention_mask = (teacher_batch[1] == 0)
        student_batch[2] = student_head_mask[layer_idx].view(1, -1, 1, 1)

        # Get the outputs of the teacher model
        with MaskNeurons(model, teacher_neuron_mask):
            layer(*teacher_batch)
        hidden_states, input_tensor = inputs.pop(0)
        teacher_output = mha_proj.dense(hidden_states) + input_tensor       # shape: [batch, seq_len, hidden_size]
        teacher_output = remove_padding(teacher_output, attention_mask)     # shape: [#tokens, hidden_size]

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            layer(*student_batch)
        hidden_states, input_tensor = inputs.pop(0)
        hidden_states = remove_padding(hidden_states, attention_mask)
        input_tensor = remove_padding(input_tensor, attention_mask)

        hidden_states = hidden_states.view(-1, num_attention_heads, attention_head_size)
        hidden_states = hidden_states.permute(1, 0, 2)

        outputs_per_head = hidden_states @ weights_per_head                 # shape: [#heads, #tokens, hidden_size]
        outputs_per_head = outputs_per_head.view(num_attention_heads, -1)   # shape: [#heads, #tokens * hidden_size]

        A = outputs_per_head.t()
        B = teacher_output - mha_proj.dense.bias - input_tensor
        B = B.flatten()

        ATA += A.t() @ A
        ATB += A.t() @ B

    handle.remove()
    return ATA, ATB


@torch.no_grad()
def get_ffn_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
):
    layer = get_layers(model)[layer_idx]
    ffn2 = get_ffn2(model, layer_idx)
    weights_per_neuron = ffn2.dense.weight.t().unsqueeze(1)                 # shape: [intermediate, 1, hidden_size]

    nonzero_neurons = student_neuron_mask[layer_idx].nonzero().squeeze()
    num_neurons = nonzero_neurons.shape[0]
    weights_per_neuron = weights_per_neuron.index_select(dim=0, index=nonzero_neurons)

    inputs = []
    handle = hijack_input(ffn2, inputs)

    ATA = torch.zeros(num_neurons, num_neurons).cuda()
    ATB = torch.zeros(num_neurons).cuda()

    model.eval()
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
        attention_mask = (teacher_batch[1] == 0)
        student_batch[2] = student_head_mask[layer_idx].view(1, -1, 1, 1)

        # Get the outputs of the teacher model
        with MaskNeurons(model, teacher_neuron_mask):
            layer(*teacher_batch)
        hidden_states, input_tensor = inputs.pop(0)
        teacher_output = ffn2.dense(hidden_states) + input_tensor           # shape: [batch, seq_len, hidden_size]
        teacher_output = remove_padding(teacher_output, attention_mask)     # shape: [#tokens, hidden_size]

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            layer(*student_batch)
        hidden_states, input_tensor = inputs.pop(0)
        hidden_states = remove_padding(hidden_states, attention_mask)
        input_tensor = remove_padding(input_tensor, attention_mask)

        hidden_states = hidden_states.t().unsqueeze(2)                      # shape: [intermediate, #tokens, 1]
        hidden_states = hidden_states.index_select(dim=0, index=nonzero_neurons)

        outputs_per_neuron = hidden_states @ weights_per_neuron             # shape: [intermediate, #tokens, hidden_size]
        outputs_per_neuron = outputs_per_neuron.view(num_neurons, -1)       # shape: [intermediate, #tokens * hidden_size]

        A = outputs_per_neuron.t()
        B = teacher_output - ffn2.dense.bias - input_tensor
        B = B.flatten()

        ATA += A.t() @ A
        ATB += A.t() @ B

    handle.remove()
    return ATA, ATB


@torch.no_grad()
def rescale(
    model,
    config,
    teacher_head_mask,
    teacher_neuron_mask,
    student_head_mask,
    student_neuron_mask,
    dataloader,
):
    num_hidden_layers = config.num_hidden_layers

    rescaled_head_mask = student_head_mask.clone()
    rescaled_neuron_mask = student_neuron_mask.clone()

    for layer_idx in tqdm(range(num_hidden_layers)):
        teacher_inputs = collect_layer_inputs(
            model,
            teacher_head_mask,
            teacher_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else teacher_inputs,
        )
        student_inputs = collect_layer_inputs(
            model,
            rescaled_head_mask,
            rescaled_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else student_inputs,
        )

        ATA, ATB = get_mha_lstsq(
            model,
            config,
            teacher_inputs,
            teacher_neuron_mask,
            student_inputs,
            rescaled_head_mask,
            rescaled_neuron_mask,
            layer_idx,
        )
        # For MHA, try to use the closed form solution as the matrix is small
        try:
            scale_factor = closed_form_solver(ATA, ATB)
        except RuntimeError:
            scale_factor = gmres_cupy_solver(ATA, ATB)
        rescaled_head_mask[layer_idx] *= scale_factor

        ATA, ATB = get_ffn_lstsq(
            model,
            config,
            teacher_inputs,
            teacher_neuron_mask,
            student_inputs,
            rescaled_head_mask,
            rescaled_neuron_mask,
            layer_idx,
        )
        # For FFN, use the GMRES solution for numerical stability
        scale_factor = gmres_cupy_solver(ATA, ATB)
        nonzero_neurons = rescaled_neuron_mask[layer_idx].nonzero().squeeze()
        for index, scale in zip(nonzero_neurons, scale_factor):
            rescaled_neuron_mask[layer_idx][index] *= scale

    return rescaled_head_mask, rescaled_neuron_mask
