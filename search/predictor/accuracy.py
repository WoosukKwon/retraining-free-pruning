import torch

from tools.glue import target_dev_metric, GLUE_TASKS
from tools.squad import post_processing_function
from tools.qa_utils import create_and_fill_np_array


class SampleAccuracyPredictor:

    def __init__(self, model, task_name, dataloader, metric, eval_dataset=None, eval_examples=None):
        self.model = model.eval()
        self.task_name = task_name
        self.eval_dataloader = dataloader
        self.metric = metric
        self.eval_dataset = eval_dataset
        self.eval_examples = eval_examples

    def predict_acc(self, configs):
        if self.task_name in GLUE_TASKS:
            return self.predict_glue_acc(configs)
        elif self.task_name in ["squad", "squad_v2"]:
            assert self.eval_examples is not None
            return self.predict_squad_acc(configs)
        else:
            raise NotImplementedError("Unsupported task")
        
    @torch.no_grad()
    def predict_glue_acc(self, configs):
        target_metric = target_dev_metric(self.task_name)
        accs =[]
        for config in configs:
            head_masks = config["head_masks"]
            filter_masks = config["filter_masks"]

            for batch in self.eval_dataloader:
                for k, v in batch.items():
                    batch[k] = v.to("cuda", non_blocking=True)

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    head_masks=head_masks,
                    filter_masks=filter_masks,
                )
                if self.model.problem_type == "regression":
                    predictions = outputs.logits.squeeze()
                else:
                    predictions = outputs.logits.argmax(dim=-1)
                self.metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )
            eval_metric = self.metric.compute()
            accuracy = eval_metric[target_metric] # FIXME
            accs.append(accuracy)

        return accs

    @torch.no_grad()
    def predict_squad_acc(self, configs):
        target_metric = "f1"
        accs = []
        for config in configs:
            head_masks = config["head_masks"]
            filter_masks = config["filter_masks"]

            all_start_logits = []
            all_end_logits = []
            for batch in self.eval_dataloader:
                for k, v in batch.items():
                    batch[k] = v.to("cuda", non_blocking=True)

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    head_masks=head_masks,
                    filter_masks=filter_masks,
                )
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                all_start_logits.append(start_logits.cpu().numpy())
                all_end_logits.append(end_logits.cpu().numpy())
            
            max_len = max([x.shape[1] for x in all_start_logits])
            start_logits_concat = create_and_fill_np_array(all_start_logits, self.eval_dataset, max_len)
            end_logits_concat = create_and_fill_np_array(all_end_logits, self.eval_dataset, max_len)
            del all_start_logits
            del all_end_logits

            outputs_numpy = (start_logits_concat, end_logits_concat)
            prediction = post_processing_function(self.task_name, self.eval_examples, self.eval_dataset, outputs_numpy)
            eval_metric = self.metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
            accuracy = eval_metric[target_metric]
            accs.append(accuracy)

        return accs