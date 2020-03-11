import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaForSequenceClassification

class InferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch['labels']

        outputs = self.model(input_ids, attention_mask=mask, labels=labels)
        
        results = {
            "loss": outputs[0],
            "outputs": outputs
        }
        return results

    def get_metrics(self, reset=False):
        pass

def get_model():
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    model = WarpModel(model)
    return model