import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

from models.common.classifier import ClassificationHead

from models.bert.config import BertConfig
from models.bert.embedding import BertEmbeddings
from models.bert.encoder import BertEncoder


class BertPretrainedModel(PreTrainedModel):

    config_class = BertConfig
    base_model_prefix = "bert"

    def _init_weights(self, module):
        return


class BertModel(BertPretrainedModel):
    
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        embedding_output = self.embeddings(input_ids=input_ids)
        attention_mask = self.get_extended_attention_mask(
            attention_mask,
            input_ids.size(),
            attention_mask.device,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
        )
        return BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
        )


class BertForSequenceClassification(BertPretrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier = ClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.is_regression:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
