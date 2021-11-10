import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class TokenPruner(nn.Module):

    def __init__(self, config, idx):
        super().__init__()

        self.threshold = nn.Parameter(torch.Tensor([idx / config.num_hidden_layers]))
        self.num_tokens = None

    def forward(self, hidden_states, attention_probs, head_masks, attention_mask, temp=None):
        if head_masks.sum() == 0.0:
            return hidden_states, attention_mask

        head_masks = head_masks.view(-1, 1)
        per_head_importance_score = attention_probs.sum(dim=2) * head_masks
        importance_score = per_head_importance_score.sum(dim=1) / head_masks.sum()
        importance_score[:, 0] += 100.0 # Keep [CLS] token

        if self.training:
            mask = torch.sigmoid((importance_score - self.threshold) / temp)
            hidden_states =  hidden_states * mask.unsqueeze(dim=2)
            self.num_tokens = (mask * (attention_mask == 0)).sum(dim=1).mean()
        else:
            indicies = torch.nonzero(importance_score > self.threshold)
            batch_size = hidden_states.shape[0]
            indicies_per_batch = []
            for i in range(batch_size):
                indicies_per_batch.append(indicies[indicies[:, 0] == i][:, 1])

            padded_indicies = pad_sequence(indicies_per_batch, batch_first=True, padding_value=0)
            hidden_states = hidden_states.gather(
                dim=1,
                index=padded_indicies.unsqueeze(2).expand(-1, -1, hidden_states.shape[2]),
            )
            attention_mask = attention_mask.gather(
                dim=3,
                index=padded_indicies.unsqueeze(1).unsqueeze(2),
            )
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask - (padded_indicies == 0) * 10000
            attention_mask[:, 0] += 10000 # [CLS] token
            attention_mask = attention_mask.view(batch_size, 1, 1, -1)

            self.num_tokens = (attention_mask == 0.0).sum() / batch_size
        return hidden_states, attention_mask
