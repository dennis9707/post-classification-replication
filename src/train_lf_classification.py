import torch
from torch import nn
from transformers import LongformerForSequenceClassification
from transformers import Trainer, TrainingArguments
from utils import compute_metrics
import argparse
import numpy as np
import pandas as pd
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
WEIGHT = None

weight_setting = {
    'api-change': [ 0.52576236, 10.20408163],
    'api-usage': [0.81699346, 1.28865979],
    'concep': [0.68306011, 1.86567164],
    'discrep': [0.72780204, 1.59744409],
    'docs': [0.51975052, 13.15789474],
    'errors': [0.64516129, 2.22222222],
    'review': [0.60386473, 2.90697674],
}

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



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

def model_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=2).to(device)

def my_hp_space(trial):
    from ray import tune

    return {
        "learning_rate": tune.loguniform(3e-5, 7e-5),
        "num_train_epochs": tune.choice(range(3, 6)),
        "seed": tune.choice(range(1, 41)),
        "per_device_train_batch_size": tune.choice([2,]),
        "per_device_eval_batch_size": tune.choice([8,]),
        "gradient_accumulation_steps": tune.choice([16, 32,]),
        # "fp16": tune.choice([True, ]),
        # "fp16_opt_level": tune.choice(['01', '02']),
    }


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
    
    batch_size = 2
    # load dataset
    train = torch.load(args.train_data)
    test = torch.load(args.test_data)
    global WEIGHT
    WEIGHT = weight_setting[args.name]
    print(args.name)
    print(WEIGHT)
    
    train.set_format("torch",
                                columns=["label", "input_ids", "attention_mask"])
    # logging_steps = len(train["train"]) // batch_size
    training_args = TrainingArguments(output_dir=args.save_folder,
                                    # num_train_epochs=3,
                                    # learning_rate=2e-5,
                                    # per_device_train_batch_size=batch_size,
                                    # per_device_eval_batch_size=batch_size,
                                    # gradient_accumulation_steps=16,
                                    # weight_decay=0.0001,
                                    fp16 = True,
                                    fp16_opt_level = '01',
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    # logging_steps=logging_steps,
                                    log_level="error")

    trainer = CustomTrainer(args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=train['train'],
                    eval_dataset=test['train'],
                    model_init=model_init, )
    trainer.hyperparameter_search(
    direction="maximize",
    hp_space=my_hp_space,
    backend="ray", 
    n_trials=10,
    # Choose among many libraries:
    # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    search_alg=HyperOptSearch(metric="objective", mode="max"),
    # Choose among schedulers:
    # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
    scheduler=ASHAScheduler(metric="objective", mode="max"))

if __name__ == '__main__':
    train()