import numpy as np


def conformal_quantile(scores, alpha):
    """Finite-sample-corrected (1-alpha) quantile of calibration nonconformity scores."""
    scores = np.asarray(scores)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))
