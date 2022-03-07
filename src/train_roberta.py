import torch
from torch import nn
from transformers import RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import argparse
import numpy as np
import pandas as pd
WEIGHT = None

weight_setting = {
    'api-change': [0.53191489, 8.33333333],
    'api-usage': [0.85034014, 1.21359223],
    'concep': [0.70422535, 1.72413793],
    'discrep': [0.67385445,1.9379845 ],
    'docs': [0.53191489, 8.33333333],
    'errors': [0.61425061, 2.68817204],
    'review': [0.59382423, 3.16455696],
}



class CustomTrainer(Trainer,):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([WEIGHT[0], WEIGHT[1]])).to(device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    weighted_precision = precision_score(labels, preds, average="weighted")
    binary_precision = precision_score(labels, preds, average="binary")
    weighted_recall = recall_score(labels, preds, average="weighted")
    binary_recall = recall_score(labels, preds, average="binary")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    binary_f1 = f1_score(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {"accuracy": acc, "weighted-precision": weighted_precision, "binary-precision": binary_precision,
            "weighted-recall": weighted_recall, "binary-recall": binary_recall,
            "weighted-f1": weighted_f1, "binary-f1": binary_f1, "auc":auc}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="api-change", type=str,
                        help="The type of the task")
    parser.add_argument("--train_data", default="../data/train/change", type=str,
                        help="The input training data file.")
    parser.add_argument("--test_data", default="../data/test/change", type=str,
                        help="The input testing data file.")
    parser.add_argument("--save_folder", default="./results/api-change", type=str,
                        help="Save folder of logs.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
    batch_size = 2
    # load dataset
    train = torch.load(args.train_data)
    test = torch.load(args.test_data)
    global WEIGHT
    WEIGHT = weight_setting[args.name]
    print("roberta-base")
    print(args.name)
    print(WEIGHT)
    
    train.set_format("torch",
                                columns=["label", "input_ids", "attention_mask"])
    logging_steps = len(train["train"]) // batch_size
    training_args = TrainingArguments(output_dir=args.save_folder,
                                    num_train_epochs=5,
                                    learning_rate=5e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    gradient_accumulation_steps=16,
                                    weight_decay=0.0001,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False,
                                    log_level="error")

    trainer = CustomTrainer(model=model, args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=train['train'],
                    eval_dataset=test['train'])
    trainer.train()

if __name__ == '__main__':
    train()