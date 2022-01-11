

def mac_per_head(
    seq_len,
    hidden_size,
    attention_head_size,
):
    per_head_qkv = lambda seq_len: 3 * seq_len * hidden_size * attention_head_size
    per_head_attn = lambda seq_len: 2 * seq_len * seq_len * attention_head_size
    per_head_output = lambda seq_len: seq_len * attention_head_size * hidden_size
    mac = per_head_qkv(seq_len) + per_head_attn(seq_len) + per_head_output(seq_len)
    return mac


def mac_per_neuron(seq_len, hidden_size):
    return 2 * seq_len * hidden_size


def compute_mac(
    num_heads_per_layer,
    num_neurons_per_layer,
    seq_len,
    hidden_size,
    attention_head_size,
):
    mac = 0.0
    for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
        attention_mac = num_heads * mac_per_head(seq_len, hidden_size, attention_head_size)
        ffn_mac = num_neurons * mac_per_neuron(seq_len, hidden_size)
        mac += attention_mac + ffn_mac
    return mac


def compute_mask_mac(head_mask, neuron_mask, seq_len, hidden_size):
    num_hidden_layers = head_mask.shape[0]
    num_attention_heads = head_mask.shape[1]
    intermediate_size = neuron_mask.shape[1]
    attention_head_size = int(hidden_size / num_attention_heads)

    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )
    pruned_mac = compute_mac(
        (head_mask != 0).sum(dim=1),
        (neuron_mask != 0).sum(dim=1),
        seq_len,
        hidden_size,
        attention_head_size,
    )
    return pruned_mac, original_mac
