import numpy as np
from sklearn.metrics import top_k_accuracy_score


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    score = top_k_accuracy_score(
        y_true, y_pred, k=4, labels=labels
    )
    return score
