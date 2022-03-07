
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    weighted_precision = precision_score(labels, preds, average="weighted")
    binary_precision = precision_score(labels, preds, average="binary")
    neg_precision = precision_score(labels, preds, pos_label = 0, average="binary")

    weighted_recall = recall_score(labels, preds, average="weighted")
    binary_recall = recall_score(labels, preds, average="binary")
    neg_recall = recall_score(labels, preds, pos_label = 0, average="binary")

    weighted_f1 = f1_score(labels, preds, average="weighted")
    binary_f1 = f1_score(labels, preds, average="binary")
    neg_f1 = f1_score(labels, preds, pos_label = 0, average="binary")
    
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {"accuracy": acc,
            "auc": auc,
            "mcc": mcc,
             
            "weighted-precision": weighted_precision, 
            "weighted-recall": weighted_recall, 
            "weighted-f1": weighted_f1, 
            
            "binary-precision": binary_precision,
            "binary-recall": binary_recall,
            "binary-f1": binary_f1,
            
            "negative-precision": neg_precision,
            "negative-recall": neg_recall, 
            "negative-f1": neg_f1, 
            }