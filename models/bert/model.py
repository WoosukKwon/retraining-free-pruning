import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
    QuestionAnsweringModelOutput,
)

from models.bert.config import BertConfig
from models.bert.embedding import BertEmbeddings
from models.bert.encoder import BertEncoder
from models.bert.pooler import BertPooler


class BertPreTrainedModel(PreTrainedModel):

    config_class = BertConfig
    base_model_prefix = "bert"

    def _init_weights(self, module):
        return


class BertModel(BertPreTrainedModel):
    
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        head_masks=None,
        filter_masks=None,
        temp=None,
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
            head_masks=head_masks,
            filter_masks=filter_masks,
            temp=temp,
        )

        last_hidden_state = encoder_outputs[0]
        if self.pooler is not None:
            pooler_output = self.pooler(last_hidden_state)
        else:
            pooler_output = None
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output= pooler_output,
        )


class BertForSequenceClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.problem_type = config.problem_type

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        head_masks=None,
        filter_masks=None,
        temp=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask,
            head_masks=head_masks,
            filter_masks=filter_masks,
            temp=temp,
        )
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class BertForQuestionAnswering(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        head_masks=None,
        filter_masks=None,
        temp=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_masks=head_masks,
            filter_masks=filter_masks,
            temp=temp,
        )
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )
