"""Mô hình đối chứng (task T40).

1. Cây quyết định tiêu chí entropy (gần với C4.5 mà hai bài CBA và CMAR so sánh). Huấn luyện trên chính
   bảng mã khoảng của CBA để so sánh công bằng (cùng thông tin đầu vào), và một bản trên thuộc tính gốc.
2. LightGBM trên thuộc tính gốc: mô hình mạnh, dùng làm "trần tham chiếu" về độ chính xác.
3. Luật một thuộc tính: dự đoán 1 khi up_rate vượt một ngưỡng chọn trên tập valid. Đây là mốc tối thiểu:
   mô hình nào không vượt được mốc này thì không đáng dùng.

Siêu tham số và ngưỡng quyết định đều chọn trên tập valid, không bao giờ trên holdout.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Ngưỡng làm cực đại F1 trên một tập (dùng với tập valid)."""
    p, r, t = precision_recall_curve(y_true, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return (float(t[i]) if len(t) else 0.5), float(f1[i])


def fit_decision_tree(X_fit: pd.DataFrame, y_fit: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray,
                      depths=(4, 6, 8, 10, 12), min_samples_leaf: int = 50, seed: int = 42):
    """Chọn max_depth theo PR-AUC trên valid. Trả về (mô hình, bảng dò tham số, thời gian huấn luyện)."""
    rows, best = [], None
    for d in depths:
        t0 = time.perf_counter()
        m = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_leaf=min_samples_leaf,
                                   random_state=seed).fit(X_fit, y_fit)
        dt = time.perf_counter() - t0
        ap = average_precision_score(y_valid, m.predict_proba(X_valid)[:, 1])
        rows.append({"max_depth": d, "pr_auc_valid": ap, "so_la": m.get_n_leaves(), "thoi_gian_s": dt})
        if best is None or ap > best[1]:
            best = (m, ap, dt)
    return best[0], pd.DataFrame(rows), best[2]


def fit_lightgbm(X_fit: pd.DataFrame, y_fit: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray,
                 seed: int = 42, params: dict | None = None):
    """LightGBM với dừng sớm theo average precision trên valid."""
    import warnings

    import lightgbm as lgb

    # LightGBM 4.7 báo eval_set sắp bị thay; vẫn dùng eval_set để chạy được trên các bản cũ hơn của Colab
    warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
    base = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_estimators=2000,
                random_state=seed, verbose=-1)
    if params:
        base.update(params)
    t0 = time.perf_counter()
    model = lgb.LGBMClassifier(**base)
    model.fit(X_fit, y_fit, eval_set=[(X_valid, y_valid)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model, time.perf_counter() - t0


class SingleFeatureRule:
    """Dự đoán 1 khi một thuộc tính vượt ngưỡng; điểm số chính là giá trị thuộc tính."""

    def __init__(self, feature: str = "up_rate"):
        self.feature = feature

    def fit(self, X_valid: pd.DataFrame, y_valid: np.ndarray) -> "SingleFeatureRule":
        self.threshold_, _ = best_f1_threshold(y_valid, X_valid[self.feature].to_numpy())
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.feature].to_numpy(dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_score(X) >= self.threshold_).astype(int)
