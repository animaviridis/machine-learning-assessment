import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn import metrics


def calculate_roc(labels_true, labels_pred):
    # Note: the ROC/ROC AUC calculation and plotting has been prepared based on:
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    classes = list(set(labels_true))
    labels_true_bin = label_binarize(labels_true, classes=classes)
    labels_pred = np.array(labels_pred)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        c = classes[i]
        y_score = labels_pred == c
        fpr[c], tpr[c], _ = metrics.roc_curve(labels_true_bin[:, i], y_score)
        roc_auc[c] = metrics.auc(fpr[c], tpr[c])

    return fpr, tpr, roc_auc


def plot_roc_curve(fpr: dict, tpr: dict, roc_auc: dict, title=None):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    for c in fpr.keys():
        ax.plot(fpr[c], tpr[c], label=f"class {c} (ROC AUC: {roc_auc[c]:.2g})", lw=2.5)
    ax.legend(fancybox=True, framealpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "ROC curves", fontsize=14)
    plt.grid()
    plt.show()


def calculate_and_plot_roc(*args, **kwargs):
    title = kwargs.pop('title', None)
    plot_roc_curve(*calculate_roc(*args, **kwargs), title=title)


def calculate_metrics(labels_true, labels_pred):
    cm = metrics.confusion_matrix(labels_true, labels_pred)
    accuracy = cm.trace() / cm.sum()
    f1_score = metrics.f1_score(labels_true, labels_pred, average='micro')

    return cm, accuracy, f1_score

