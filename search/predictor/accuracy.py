import torch

from tools.glue import target_dev_metric


class SampleAccuracyPredictor:

    def __init__(self, model, task_name, dataloader, metric):
        self.model = model.eval()
        self.task_name = task_name
        self.eval_dataloader = dataloader
        self.metric = metric
        self.target_metric = target_dev_metric(self.task_name)

    @torch.no_grad()
    def predict_acc(self, configs):
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
            accuracy = eval_metric[self.target_metric] # FIXME
            accs.append(accuracy)
        return accs
