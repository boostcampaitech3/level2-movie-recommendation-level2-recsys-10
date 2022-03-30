from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import logging

def evaluate_metrics(y_true, y_pred, metrics, **kwargs):
    result = dict()
    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            result[metric] = log_loss(y_true, y_pred, eps=1e-7)
        elif metric == 'AUC':
            result[metric] = roc_auc_score(y_true, y_pred)
        elif metric == "ACC":
            y_pred = np.argmax(y_pred, axis=1)
            result[metric] = accuracy_score(y_true, y_pred)
        else:
            assert "group_index" in kwargs, "group_index is required for GAUC"
            group_index = kwargs["group_index"]
            if metric == "GAUC":
                pass
            elif metric == "NDCG":
                pass
            elif metric == "MRR":
                pass
            elif metric == "HitRate":
                pass
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
    return result