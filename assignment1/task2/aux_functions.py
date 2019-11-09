import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV
from tqdm import tqdm
import itertools

from decisiontree import Node


logger = logging.getLogger(__name__)


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
    ax.set_aspect('equal')
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

    return dict(cm=cm, accuracy=accuracy, f1_score=f1_score)


def cross_validate_tree(n_splits, data, **kwargs):
    logger.info(f"Decision tree learning and testing with {n_splits}-fold cross validation")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    test_labels_true = []
    test_labels_pred = []

    for i, (train_idx, test_idx) in tqdm(enumerate(splitter.split(data))):
        logger.info(f"Cross-validation round {i} with {len(train_idx)} train samples and {len(test_idx)} test samples")
        logger.debug(f"Test indices: {test_idx}")

        true_i, pred_i = Node.train_and_test(data, train_idx, test_idx, **kwargs)
        test_labels_true.extend(true_i)
        test_labels_pred.extend(pred_i)

    calculate_and_plot_roc(test_labels_true, test_labels_pred, title="ROC curves for wine data classification")
    return calculate_metrics(test_labels_true, test_labels_pred)


def cross_validate_sklearn(estimator, n_splits, data_x, data_y):
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    test_labels_true = []
    test_labels_pred = []

    for i, (train_idx, test_idx) in enumerate(splitter.split(data_x)):
        logger.debug(f"Cross-validation round {i} with {len(train_idx)} train samples and {len(test_idx)} test samples")
        logger.debug(f"Test indices: {test_idx}")

        test_labels_true.extend(data_y[test_idx])
        test_labels_pred.extend(estimator.fit(data_x.loc[train_idx], data_y[train_idx]).predict(data_x.loc[test_idx]))

    calculate_and_plot_roc(test_labels_true, test_labels_pred, title="ROC curves for wine data classification")
    return calculate_metrics(test_labels_true, test_labels_pred)


def tune_params(func, params, func_args=(), func_kwargs=None, scoring_metrics='metrics'):
    func_kwargs = func_kwargs or {}
    results = []

    params_keys = params.keys()
    params_prod = itertools.product(*params.values())

    for params_i in tqdm(params_prod):
        params_i_dict = dict(zip(params_keys, params_i))
        xv = func(*func_args, **func_kwargs, **params_i_dict)
        params_i_dict.update(xv if isinstance(xv, dict) else {'metrics': xv})
        results.append(params_i_dict)

    best_result = results[int(np.argmax([t[scoring_metrics] for t in results]))]
    return results, best_result


def make_grid_searcher(data_x, data_y, n_splits=10):
    def wrapper(estimator, params):
        gscv = GridSearchCV(estimator, params, cv=n_splits)
        res = gscv.fit(data_x, data_y)
        res_stats = cross_validate_sklearn(res.best_estimator_, n_splits=n_splits, data_x=data_x, data_y=data_y)
        return res.best_params_, res_stats
    return wrapper
