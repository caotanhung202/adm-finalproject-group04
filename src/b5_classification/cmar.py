"""CMAR: Classification based on Multiple Association Rules (Li, Han và Pei, 2001).

Cài đặt bám theo bài báo trong thư mục Project ("7. Data classification.pdf"):

Sinh luật: mọi luật P -> c có support >= minsup và confidence >= minconf. Bài báo dùng biến thể FP-growth
và cây CR-tree để tăng tốc và tiết kiệm bộ nhớ; hai cấu trúc này không làm thay đổi tập luật thu được,
nên ở đây dùng chung bộ đếm condset với CBA (cba.count_condsets).

Thứ tự hạng (rank): R1 > R2 nếu (1) conf(R1) > conf(R2); hoặc (2) bằng confidence nhưng sup(R1) > sup(R2);
hoặc (3) bằng cả hai nhưng R1 có ít điều kiện hơn.

Ba bước cắt tỉa (mục 3.3 của bài báo):
    1. Luật tổng quát: R1: P -> c là luật tổng quát của R2: P' -> c' nếu P là tập con của P'. Bỏ R2 nếu có
       R1 tổng quát của nó và R1 có hạng cao hơn. (Bài báo không yêu cầu c = c'; cài đặt theo đúng chữ.)
    2. Chỉ giữ luật tương quan dương có ý nghĩa thống kê theo kiểm định chi bình phương (bảng 2x2 giữa
       "khớp P" và "thuộc lớp c"). Bài báo không nêu mức ý nghĩa; nhóm chọn alpha = 5% (ngưỡng 3,841).
    3. Độ phủ cơ sở dữ liệu với ngưỡng delta (bài báo dùng 4): duyệt luật theo hạng, chọn luật nếu nó phân
       lớp đúng ít nhất một dòng còn lại; mỗi dòng khớp luật được chọn tăng bộ đếm phủ thêm 1; dòng bị loại
       khi bộ đếm phủ đạt delta.

Phân lớp (mục 4): lấy mọi luật khớp với dòng. Nếu cùng một lớp thì gán lớp đó. Nếu khác lớp, chia nhóm
theo lớp và chọn nhóm có tổng chi bình phương có trọng số lớn nhất:
        sum( chi2(R) * chi2(R) / maxchi2(R) )
    với maxchi2 = (min(sup(P), sup(c)) - sup(P) * sup(c) / |T|)^2 * |T| * e,
        e = 1/(sup(P) sup(c)) + 1/(sup(P)(|T| - sup(c))) + 1/((|T| - sup(P)) sup(c))
            + 1/((|T| - sup(P))(|T| - sup(c))).
    Bài báo không nói trường hợp không luật nào khớp; nhóm gán lớp đa số của dữ liệu huấn luyện.

Ghi chú về "ngưỡng chênh lệch confidence 20%": bài báo nêu giá trị này trong phần thực nghiệm và cho biết
nó cùng ngưỡng độ phủ quyết định số luật dùng để phân lớp, nhưng không mô tả cơ chế cụ thể (dẫn sang luận
văn thạc sĩ của W. Li, 2001). Vì không có định nghĩa trong bài báo, cài đặt này KHÔNG dùng ngưỡng đó và
ghi rõ là một điểm khác biệt.
"""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd

from .cba import CondsetTable, class_min_counts, count_condsets, encode_codes

CHI2_95 = 3.841458820694124


def chi2_2x2(sup_p: np.ndarray, sup_c: np.ndarray, sup_pc: np.ndarray, n: float) -> np.ndarray:
    """Chi bình phương của bảng 2x2 (khớp P hay không) x (lớp c hay không), không hiệu chỉnh Yates."""
    sup_p = sup_p.astype(float)
    sup_c = sup_c.astype(float)
    a = sup_pc.astype(float)                    # P và c
    b = sup_p - a                               # P và không c
    c = sup_c - a                               # không P và c
    d = n - sup_p - sup_c + a                   # không P và không c
    num = n * (a * d - b * c) ** 2
    den = sup_p * (n - sup_p) * sup_c * (n - sup_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def max_chi2(sup_p: np.ndarray, sup_c: np.ndarray, n: float) -> np.ndarray:
    """Cận trên của chi bình phương theo công thức trong bài CMAR."""
    sp = sup_p.astype(float)
    sc = sup_c.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = 1 / (sp * sc) + 1 / (sp * (n - sc)) + 1 / ((n - sp) * sc) + 1 / ((n - sp) * (n - sc))
        out = (np.minimum(sp, sc) - sp * sc / n) ** 2 * n * e
    return np.where(np.isfinite(out) & (out > 0), out, np.nan)


class CMARClassifier:
    """Bộ phân lớp CMAR (minsup 1%, minconf 50%, delta = 4 theo phần thực nghiệm của bài báo)."""

    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5, max_len: int = 3,
                 coverage_delta: int = 4, chi2_threshold: float = CHI2_95, pos_label: int = 1,
                 class_minsup: bool = False):
        self.class_minsup = class_minsup
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.coverage_delta = coverage_delta
        self.chi2_threshold = chi2_threshold
        self.pos_label = pos_label

    def _candidate_rules(self, table: CondsetTable, min_count) -> pd.DataFrame:
        cond_sup = table.counts.sum(axis=1)
        rows = []
        for c in range(table.counts.shape[1]):
            rs = table.counts[:, c]
            conf = rs / np.maximum(cond_sup, 1)
            thr = np.broadcast_to(np.asarray(min_count, dtype=float), (table.counts.shape[1],))[c]
            ok = (rs >= thr) & (conf >= self.min_confidence)
            for i in np.where(ok)[0]:
                rows.append((i, c, cond_sup[i], rs[i], conf[i], len(table.condsets[i])))
        df = pd.DataFrame(rows, columns=["cid", "cls_idx", "condsup", "rulesup", "confidence", "length"])
        df["support"] = df["rulesup"] / table.n
        # Hạng: confidence giảm, support giảm, ít điều kiện hơn đứng trước
        return df.sort_values(["confidence", "rulesup", "length"], ascending=[False, False, True]).reset_index(drop=True)

    def _prune_general(self, rules: pd.DataFrame, table: CondsetTable) -> pd.DataFrame:
        kept_condsets: set = set()
        keep = np.zeros(len(rules), dtype=bool)
        for j, cid in enumerate(rules["cid"]):
            cs = table.condsets[cid]
            general_exists = any(sub in kept_condsets
                                 for k in range(1, len(cs)) for sub in combinations(cs, k))
            if general_exists:
                continue
            keep[j] = True
            kept_condsets.add(cs)
        return rules[keep].reset_index(drop=True)

    def _prune_chi2(self, rules: pd.DataFrame, table: CondsetTable) -> pd.DataFrame:
        sup_c = table.class_totals[rules["cls_idx"].to_numpy()]
        chi = chi2_2x2(rules["condsup"].to_numpy(), sup_c, rules["rulesup"].to_numpy(), table.n)
        expected = rules["condsup"].to_numpy() * sup_c / table.n
        positive = rules["rulesup"].to_numpy() > expected
        rules = rules.assign(chi2=chi, max_chi2=max_chi2(rules["condsup"].to_numpy(), sup_c, table.n))
        return rules[(chi >= self.chi2_threshold) & positive].reset_index(drop=True)

    def _prune_coverage(self, rules: pd.DataFrame, table: CondsetTable, codes: np.ndarray,
                        y_idx: np.ndarray) -> pd.DataFrame:
        cover = np.zeros(len(y_idx), dtype=np.int32)
        alive = np.ones(len(y_idx), dtype=bool)
        keep = np.zeros(len(rules), dtype=bool)
        for j, r in enumerate(rules.itertuples(index=False)):
            if not alive.any():
                break
            m = alive.copy()
            for a, v in table.condsets[r.cid]:
                m &= codes[:, a] == v
            if not np.any(m & (y_idx == r.cls_idx)):
                continue
            keep[j] = True
            cover[m] += 1
            alive &= cover < self.coverage_delta
        return rules[keep].reset_index(drop=True)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CMARClassifier":
        t0 = time.perf_counter()
        codes, self.feature_names_, cards = encode_codes(X)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        class_totals = np.bincount(y_idx, minlength=len(self.classes_))
        min_count = class_min_counts(self.min_support, class_totals, len(y_idx), self.class_minsup)
        table = count_condsets(codes, cards, y_idx, len(self.classes_), min_count, self.max_len)
        rules = self._candidate_rules(table, min_count)
        self.n_candidate_rules_ = len(rules)
        rules = self._prune_general(rules, table)
        self.n_after_general_ = len(rules)
        rules = self._prune_chi2(rules, table)
        self.n_after_chi2_ = len(rules)
        rules = self._prune_coverage(rules, table, codes, y_idx)
        self.n_rules_ = len(rules)
        rules["items"] = [table.condsets[c] for c in rules["cid"]]
        rules["weight"] = rules["chi2"] ** 2 / rules["max_chi2"]
        self.rules_ = rules
        self.default_idx_ = int(table.class_totals.argmax())
        self.pos_idx_ = int(np.where(self.classes_ == self.pos_label)[0][0]) if self.pos_label in self.classes_ else 0
        self.default_score_ = table.class_totals[self.pos_idx_] / table.class_totals.sum()
        self.table_ = table
        self.timing_ = {"total_fit_s": time.perf_counter() - t0}
        return self

    def _group_weights(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        codes = X[self.feature_names_].to_numpy(dtype=np.int64) + 1
        n, n_cls = len(codes), len(self.classes_)
        W = np.zeros((n, n_cls))
        n_match = np.zeros(n, dtype=np.int32)
        for r in self.rules_.itertuples(index=False):
            m = np.ones(n, dtype=bool)
            for a, v in r.items:
                m &= codes[:, a] == v
            w = r.weight if np.isfinite(r.weight) else 0.0
            W[m, r.cls_idx] += w
            n_match += m
        return W, n_match

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        W, n_match = self._group_weights(X)
        pred = W.argmax(axis=1)
        pred[n_match == 0] = self.default_idx_
        return self.classes_[pred]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Điểm cho lớp dương = tỷ trọng chi bình phương có trọng số của nhóm lớp dương (nhóm bổ sung)."""
        W, n_match = self._group_weights(X)
        tot = W.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(tot > 0, W[:, self.pos_idx_] / tot, self.default_score_)
        s[n_match == 0] = self.default_score_
        return s

    def rules_table(self, labelers: dict | None = None, describe=None) -> pd.DataFrame:
        out = self.rules_[["confidence", "support", "chi2", "weight", "cls_idx"]].copy()
        out["lop"] = self.classes_[out.pop("cls_idx").to_numpy()]
        if describe is not None:
            out.insert(0, "dieu_kien", [describe(it, labelers) for it in self.rules_["items"]])
        return out
