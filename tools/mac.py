

def compute_mac(
    num_heads_per_layer,
    num_neurons_per_layer,
    seq_len_per_layer,
    hidden_size,
    attention_head_size,
):
    per_head_qkv = lambda seq_len: 3 * seq_len * hidden_size * attention_head_size
    per_head_attn = lambda seq_len: 2 * seq_len * seq_len * attention_head_size
    per_head_output = lambda seq_len: seq_len * attention_head_size * hidden_size

    num_hidden_layers = len(num_heads_per_layer)

    if isinstance(seq_len_per_layer, (int, float)):
        seq_len_per_layer = [seq_len_per_layer] * (num_hidden_layers + 1)
    assert len(seq_len_per_layer) == num_hidden_layers + 1
    
    mac = 0.0
    for i in range(num_hidden_layers):
        num_heads = num_heads_per_layer[i]
        num_neurons = num_neurons_per_layer[i]
        seq_len = seq_len_per_layer[i]

        mac_per_head = per_head_qkv(seq_len) + per_head_attn(seq_len) + per_head_output(seq_len)
        attention_mac = num_heads * mac_per_head

        next_seq_len = seq_len_per_layer[i + 1]
        ffn_mac = 2 * next_seq_len * hidden_size * num_neurons
        mac += attention_mac + ffn_mac

    return mac # FIXME: add classifier mac
