import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def metrics(y, pred, score):
    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)

    roc_auc = roc_auc_score(y, score[:, 1])  
    average_precision = average_precision_score(y, score[:, 1])

    metrics_dict = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1,
                    "AUC": roc_auc, "AUPRC": average_precision}
    return metrics_dict


def label2index(label):
    label_dict = {'Non-': 0, 'Eff-': 1}
    index = label_dict[label]
    return index


def viz_conf_matrix(cm, labels, figsize=(7, 6)):
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if p < 1:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    return fig
