"""Độ đo, khoảng tin cậy bootstrap theo người dùng, phân tích lỗi và biểu đồ (task T42, T43).

Vì sao bootstrap theo người dùng: các cặp của cùng một người dùng không độc lập với nhau (cùng thói quen,
cùng một đơn train). Lấy mẫu lại theo dòng sẽ làm khoảng tin cậy hẹp giả tạo; lấy mẫu lại theo người dùng
(cluster bootstrap) tôn trọng cấu trúc này.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def classification_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "ty_le_du_doan_1": float(np.mean(pred)),
    }
    if score is not None and len(np.unique(y)) > 1:
        out["pr_auc"] = average_precision_score(y, score)
        out["roc_auc"] = roc_auc_score(y, score)
    return out


def _user_groups(users: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(users, kind="stable")
    u_sorted = users[order]
    cuts = np.flatnonzero(np.diff(u_sorted)) + 1
    return np.split(order, cuts)


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, score: np.ndarray, users: np.ndarray,
                 n_boot: int = 200, seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """Khoảng tin cậy 95% cho F1 và PR-AUC bằng bootstrap theo người dùng."""
    groups = _user_groups(np.asarray(users))
    rng = np.random.default_rng(seed)
    f1s, aps = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        f1s.append(f1_score(y[idx], pred[idx], zero_division=0))
        if len(np.unique(y[idx])) > 1:
            aps.append(average_precision_score(y[idx], score[idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return pd.DataFrame({"do_do": ["f1", "pr_auc"],
                         "can_duoi": [np.percentile(f1s, lo), np.percentile(aps, lo)],
                         "can_tren": [np.percentile(f1s, hi), np.percentile(aps, hi)]})


def paired_bootstrap_diff(y: np.ndarray, users: np.ndarray, a: dict, b: dict, n_boot: int = 200,
                          seed: int = 42) -> pd.DataFrame:
    """Chênh lệch (A trừ B) của F1 và PR-AUC, cùng một mẫu người dùng cho hai mô hình ở mỗi lần lặp.

    a, b: dict có khóa 'pred' và 'score'. Nếu khoảng 95% không chứa 0 thì khác biệt có ý nghĩa.
    """
    groups = _user_groups(np.asarray(users))
    rng = np.random.default_rng(seed)
    d_f1, d_ap = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        yy = y[idx]
        d_f1.append(f1_score(yy, a["pred"][idx], zero_division=0) - f1_score(yy, b["pred"][idx], zero_division=0))
        if len(np.unique(yy)) > 1:
            d_ap.append(average_precision_score(yy, a["score"][idx]) - average_precision_score(yy, b["score"][idx]))
    return pd.DataFrame({"do_do": ["f1", "pr_auc"],
                         "chenh_lech_trung_vi": [np.median(d_f1), np.median(d_ap)],
                         "can_duoi_95": [np.percentile(d_f1, 2.5), np.percentile(d_ap, 2.5)],
                         "can_tren_95": [np.percentile(d_f1, 97.5), np.percentile(d_ap, 97.5)]})


def error_by_group(y: np.ndarray, pred: np.ndarray, group: pd.Series, name: str) -> pd.DataFrame:
    """Precision, recall, F1 và cơ cấu lỗi theo từng nhóm (ví dụ nhóm số đơn, ngành hàng)."""
    df = pd.DataFrame({"y": y, "p": pred, "g": pd.Series(group).reset_index(drop=True)})
    rows = []
    for g, d in df.groupby("g", observed=True):
        tp = int(((d.y == 1) & (d.p == 1)).sum())
        fp = int(((d.y == 0) & (d.p == 1)).sum())
        fn = int(((d.y == 1) & (d.p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else np.nan
        rows.append({name: g, "so_cap": len(d), "ty_le_nhan_1": d.y.mean(), "precision": prec,
                     "recall": rec, "f1": f1, "duong_tinh_gia": fp, "am_tinh_gia": fn})
    return pd.DataFrame(rows)


def plot_pr_curves(y: np.ndarray, scores: dict, path: str | None = None, title: str = ""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{name} (PR-AUC = {average_precision_score(y, s):.3f})".replace(".", ","), lw=1.6)
    ax.axhline(np.mean(y), ls="--", color="grey", lw=1, label=f"Tỷ lệ nhãn 1 = {np.mean(y):.3f}".replace(".", ","))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def plot_confusion(y: np.ndarray, preds: dict, path: str | None = None):
    import matplotlib.pyplot as plt

    k = len(preds)
    fig, axes = plt.subplots(1, k, figsize=(3.4 * k, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (name, p) in zip(axes, preds.items()):
        cm = confusion_matrix(y, p, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["Dự đoán 0", "Dự đoán 1"], fontsize=8)
        ax.set_yticks([0, 1], ["Thực tế 0", "Thực tế 1"], fontsize=8)
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def format_vn(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Định dạng số theo quy ước của nhóm: dấu chấm hàng nghìn, dấu phẩy thập phân (chỉ để hiển thị)."""
    def f(v):
        if isinstance(v, (int, np.integer)):
            return f"{v:,}".replace(",", ".")
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            s = f"{v:,.{decimals}f}"
            return s.replace(",", "X").replace(".", ",").replace("X", ".")
        return v
    return df.map(f)
