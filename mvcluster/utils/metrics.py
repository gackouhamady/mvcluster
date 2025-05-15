import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    return conf_mat[row_ind, :][:, col_ind]

def cmat_to_psuedo_y_true_and_y_pred(cmat):
    y_true, y_pred = [], []
    for true_class, row in enumerate(cmat):
        for pred_class, val in enumerate(row):
            y_true.extend([true_class] * val)
            y_pred.extend([pred_class] * val)
    return y_true, y_pred

def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)

def clustering_f1_score(y_true, y_pred, **kwargs):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    pseudo_y_true, pseudo_y_pred = cmat_to_psuedo_y_true_and_y_pred(conf_mat)
    return metrics.f1_score(pseudo_y_true, pseudo_y_pred, **kwargs)
