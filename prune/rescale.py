import torch
import cupy
from cupyx.scipy.sparse.linalg import gmres

from utils.arch import (
    get_encoder,
    get_layers,
    get_mha_proj,
    get_ffn2,
    hijack_input,
    MaskNeurons,
    remove_padding,
)


def collect_layer_inputs(
    model,
    head_mask,
    neuron_mask,
    layer_idx,
    dataloader=None,
    prev_inputs=None,
):
    layers = get_layers(model)
    target_layer = layers[layer_idx]

    inputs = []
    if layer_idx == 0:
        assert dataloader is not None
        encoder = get_encoder(model)
        layers = encoder.layer
        encoder.layers = layers[:1]

        handle = hijack_input(target_layer, inputs)
        for batch in dataloader:
            for k, v in batch.items():
                batch[k] = v.to("cuda")
            with MaskNeurons(model, neuron_mask):
                model(head_mask=head_mask, **batch)

        handle.remove()
        encoder.layers = layers
    else:
        assert prev_inputs is not None
        prev_layer = layers[layer_idx - 1]

        for batch in prev_inputs:
            batch[2] = head_mask[layer_idx - 1]
            with MaskNeurons(model, neuron_mask):
                prev_output = prev_layer(*batch)

            batch[0] = prev_output[0]
            batch[2] = head_mask[layer_idx]
            inputs.append(batch)

    return inputs


@torch.no_grad()
def get_mha_lstsq(
    model,
    config,
    teacher_head_mask,
    teacher_neuron_mask,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
    dataloader,
):
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    mha_proj = get_mha_proj(model, layer_idx)
    weights_per_head = mha_proj.dense.weight.t().view(num_attention_heads, -1, hidden_size)

    inputs = []
    handle = hijack_input(mha_proj, inputs)

    ATA = torch.zeros(num_attention_heads, num_attention_heads).cuda()
    ATB = torch.zeros(num_attention_heads).cuda()

    encoder = get_encoder(model)
    layers = encoder.layer
    encoder.layers = layers[:layer_idx + 1]

    model.eval()
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to("cuda")
        attention_mask = batch["attention_mask"]

        # Get the outputs of the teacher model
        with MaskNeurons(model, teacher_neuron_mask):
            model(head_mask=teacher_head_mask, **batch)
        hidden_states, input_tensor = inputs.pop(0)
        teacher_output = mha_proj.dense(hidden_states) + input_tensor       # shape: [batch, seq_len, hidden_size]
        teacher_output = remove_padding(teacher_output, attention_mask)     # shape: [#tokens, hidden_size]

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            model(head_mask=student_head_mask, **batch)
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
    encoder.layers = layers
    return ATA, ATB


@torch.no_grad()
def get_ffn_lstsq(
    model,
    config,
    teacher_head_mask,
    teacher_neuron_mask,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
    dataloader,
):
    intermediate_size = config.intermediate_size

    ffn2 = get_ffn2(model, layer_idx)
    weights_per_neuron = ffn2.dense.weight.t().unsqueeze(1)                 # shape: [intermediate, 1, hidden_size]

    inputs = []
    handle = hijack_input(ffn2, inputs)

    ATA = torch.zeros(intermediate_size, intermediate_size).cuda()
    ATB = torch.zeros(intermediate_size).cuda()

    encoder = get_encoder(model)
    layers = encoder.layer
    encoder.layers = layers[:layer_idx + 1]

    model.eval()
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to("cuda")
        attention_mask = batch["attention_mask"]

        # Get the outputs of the teacher model
        with MaskNeurons(model, teacher_neuron_mask):
            model(head_mask=teacher_head_mask, **batch)
        hidden_states, input_tensor = inputs.pop(0)
        teacher_output = ffn2.dense(hidden_states) + input_tensor           # shape: [batch, seq_len, hidden_size]
        teacher_output = remove_padding(teacher_output, attention_mask)     # shape: [#tokens, hidden_size]

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            model(head_mask=student_head_mask, **batch)
        hidden_states, input_tensor = inputs.pop(0)
        hidden_states = remove_padding(hidden_states, attention_mask)
        input_tensor = remove_padding(input_tensor, attention_mask)

        hidden_states = hidden_states.t().unsqueeze(2)                      # shape: [intermediate, #tokens, 1]

        outputs_per_neuron = hidden_states @ weights_per_neuron             # shape: [intermediate, #tokens, hidden_size]
        outputs_per_neuron = outputs_per_neuron.view(intermediate_size, -1) # shape: [intermediate, #tokens * hidden_size]

        A = outputs_per_neuron.t()
        B = teacher_output - ffn2.dense.bias - input_tensor
        B = B.flatten()

        ATA += A.t() @ A
        ATB += A.t() @ B

    handle.remove()
    encoder.layers = layers
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

    rescaled_head_mask = torch.zeros_like(student_head_mask)
    rescaled_neuron_mask = torch.zeros_like(student_neuron_mask)

    for layer_idx in range(num_hidden_layers):
        ATA, ATB = get_mha_lstsq(
            model,
            config,
            teacher_head_mask,
            teacher_neuron_mask,
            student_head_mask,
            student_neuron_mask,
            layer_idx,
            dataloader,
        )
        # For MHA, use the closed form solution as the matrix is small
        scale_factor = torch.inverse(ATA) @ ATB
        print(scale_factor)
        rescaled_head_mask[layer_idx] = scale_factor * student_head_mask[layer_idx]

        ATA, ATB = get_ffn_lstsq(
            model,
            config,
            teacher_head_mask,
            teacher_neuron_mask,
            student_head_mask,
            student_neuron_mask,
            layer_idx,
            dataloader,
        )
        # For FFN, use the GMRES solution for numerical stability
        CU_ATA = cupy.asarray(ATA.cpu().numpy())
        CU_ATB = cupy.asarray(ATB.cpu().numpy())
        solution = gmres(CU_ATA, CU_ATB)
        scale_factor = cupy.asnumpy(solution[0])
        scale_factor = torch.from_numpy(scale_factor).cuda()
        rescaled_neuron_mask = scale_factor * student_neuron_mask[layer_idx]

    return rescaled_head_mask, rescaled_neuron_mask
