"""CBA: Classification Based on Associations (Liu, Hsu và Ma, 1998).

Cài đặt bám theo bài báo trong thư mục Project ("6. Data classification.pdf"):

CBA-RG (sinh luật lớp)
    - Mỗi item là một cặp (thuộc tính, mã khoảng). Ruleitem có dạng <condset, y>.
    - condsupCount = số dòng chứa condset; rulesupCount = số dòng chứa condset và có lớp y.
    - support = rulesupCount / |D|; confidence = rulesupCount / condsupCount.
    - Với mỗi condset, chọn ruleitem có confidence cao nhất làm "luật có thể" (PR). Luật lớp (CAR) là các
      PR vừa phổ biến (support >= minsup) vừa chính xác (confidence >= minconf).
    - Cắt tỉa tùy chọn (pruneRules): bỏ luật r nếu tỷ lệ lỗi bi quan của r cao hơn tỷ lệ lỗi bi quan của
      một luật r- có ít hơn một điều kiện (phương pháp của C4.5, mức tin cậy CF = 25%).

    Cách đếm: bài báo dùng Apriori sinh ứng viên theo từng mức. Vì mỗi dòng có đúng một giá trị cho mỗi
    thuộc tính, ở đây đếm trực tiếp mọi condset của từng tổ hợp thuộc tính (tối đa max_len thuộc tính)
    bằng np.bincount. Tập ruleitem phổ biến thu được giống hệt Apriori (tính chất chống đơn điệu chỉ giúp
    Apriori bỏ bớt ứng viên, không làm thay đổi kết quả); đổi lại cách đếm này nhanh trên dữ liệu nằm
    trong bộ nhớ.

CBA-CB (dựng bộ phân lớp, phương án M1 trong Hình 2 của bài báo)
    1. Sắp luật theo thứ tự ưu tiên: confidence giảm dần, rồi support giảm dần, rồi luật sinh trước
       (condset ngắn hơn được sinh trước) đứng trước.
    2. Duyệt luật theo thứ tự: luật được giữ nếu phân lớp đúng ít nhất một dòng còn lại; xóa mọi dòng nó
       phủ; chọn lớp mặc định là lớp đa số của phần còn lại; ghi tổng lỗi = lỗi các luật đã chọn + lỗi
       của lớp mặc định.
    3. Cắt danh sách tại luật đầu tiên có tổng lỗi nhỏ nhất; lớp mặc định đi kèm luật đó là lớp mặc định
       của bộ phân lớp.

    Bài báo đề xuất M2 để giảm số lượt quét khi dữ liệu nằm trên đĩa. Dữ liệu của đồ án nằm trọn trong
    bộ nhớ, và M1 được viết dạng vectơ (mỗi luật một phép AND trên mảng bool), nên dùng M1 cho minh bạch.

Phân lớp: luật đầu tiên (theo thứ tự ưu tiên) khớp với dòng sẽ quyết định lớp; không luật nào khớp thì
dùng lớp mặc định.

Điểm số để vẽ đường precision theo recall (không có trong bài báo, do nhóm bổ sung): điểm của một dòng là
tỷ lệ lớp dương trong các dòng huấn luyện khớp condset của luật quyết định dòng đó; dòng rơi vào lớp mặc
định nhận tỷ lệ lớp dương của phần dữ liệu còn lại tại điểm cắt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import beta

Item = tuple[int, int]          # (chỉ số thuộc tính, mã khoảng đã dịch +1)
Condset = tuple[Item, ...]


# ---------------------------------------------------------------------------
# Phần dùng chung với CMAR: đếm mọi condset có ít nhất một ruleitem phổ biến
# ---------------------------------------------------------------------------
@dataclass
class CondsetTable:
    """Kết quả đếm: mỗi condset kèm vectơ số dòng theo lớp và thứ tự sinh."""

    condsets: list[Condset]
    counts: np.ndarray            # shape (số condset, số lớp)
    gen_order: np.ndarray         # thứ tự sinh (dùng để phá hòa)
    class_totals: np.ndarray      # số dòng mỗi lớp trên toàn bộ dữ liệu huấn luyện
    n: int
    index: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {c: i for i, c in enumerate(self.condsets)}

    def counts_of(self, condset: Condset) -> np.ndarray | None:
        if len(condset) == 0:
            return self.class_totals
        i = self.index.get(condset)
        return None if i is None else self.counts[i]


def encode_codes(X: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Đổi bảng mã khoảng (có -1 cho giá trị thiếu) thành ma trận số nguyên không âm."""
    codes = X.to_numpy(dtype=np.int64) + 1
    if (codes < 0).any():
        raise ValueError("Mã khoảng phải >= -1.")
    cards = codes.max(axis=0) + 1
    return codes, list(X.columns), cards


def count_condsets(codes: np.ndarray, cards: np.ndarray, y_idx: np.ndarray, n_classes: int,
                   min_count, max_len: int) -> CondsetTable:
    """Đếm mọi condset gồm 1..max_len thuộc tính khác nhau; giữ condset có ít nhất một lớp đạt ngưỡng.

    min_count là một số (ngưỡng chung) hoặc một mảng ngưỡng theo từng lớp.
    """
    n, m = codes.shape
    condsets: list[Condset] = []
    counts_list = []
    class_totals = np.bincount(y_idx, minlength=n_classes)
    # Một tổ hợp thuộc tính chỉ cần xét nếu mọi tổ hợp con (bớt một thuộc tính) có condset phổ biến;
    # đây chính là ý tưởng cắt tỉa của Apriori áp dụng ở mức tổ hợp thuộc tính.
    alive_combos = {(): True}
    for k in range(1, max_len + 1):
        any_alive = False
        for combo in combinations(range(m), k):
            if k > 1 and not all(alive_combos.get(sub, False) for sub in combinations(combo, k - 1)):
                continue
            key = np.zeros(n, dtype=np.int64)
            mult = 1
            for a in combo:
                key += codes[:, a] * mult
                mult *= int(cards[a])
            cnt = np.bincount(key * n_classes + y_idx, minlength=mult * n_classes).reshape(mult, n_classes)
            frequent = np.where((cnt >= min_count).any(axis=1))[0]
            alive_combos[combo] = len(frequent) > 0
            any_alive |= len(frequent) > 0
            for kv in frequent:
                rem = int(kv)
                items = []
                for a in combo:
                    items.append((a, rem % int(cards[a])))
                    rem //= int(cards[a])
                condsets.append(tuple(items))
                counts_list.append(cnt[kv])
        if not any_alive:
            break
    counts = np.vstack(counts_list) if counts_list else np.zeros((0, n_classes), dtype=np.int64)
    return CondsetTable(condsets, counts, np.arange(len(condsets)), class_totals, n)


def class_min_counts(min_support: float, class_totals: np.ndarray, n: int, class_minsup: bool) -> np.ndarray:
    """Ngưỡng đếm tối thiểu cho từng lớp.

    class_minsup = False: một ngưỡng chung minsup × |D| như bài CBA gốc (Liu et al., 1998).
    class_minsup = True : minsup riêng cho lớp c bằng minsup × tỷ lệ của lớp c, tức ngưỡng đếm
                          minsup × (số dòng lớp c). Đây là ý tưởng "nhiều ngưỡng hỗ trợ theo lớp" của
                          CBA cải tiến (Liu, Ma & Wong, 2000), dùng khi lớp phân bố lệch: với ngưỡng chung,
                          lớp hiếm gần như không có luật nào đạt ngưỡng.
    """
    if class_minsup:
        return min_support * class_totals.astype(float)
    return np.full(len(class_totals), min_support * n, dtype=float)


def pessimistic_error(errors: np.ndarray, total: np.ndarray, cf: float = 0.25) -> np.ndarray:
    """Cận trên tỷ lệ lỗi theo phân phối nhị thức ở mức tin cậy CF (ý tưởng pessimistic error của C4.5)."""
    errors = np.asarray(errors, dtype=float)
    total = np.asarray(total, dtype=float)
    out = np.ones_like(errors)
    ok = errors < total
    out[ok] = beta.ppf(1 - cf, errors[ok] + 1, total[ok] - errors[ok])
    return out


# ---------------------------------------------------------------------------
# CBA
# ---------------------------------------------------------------------------
class CBAClassifier:
    """Bộ phân lớp CBA.

    Tham số
        min_support    : minsup tính theo tỷ lệ trên dữ liệu huấn luyện (bài báo dùng 1%)
        min_confidence : minconf (bài báo dùng 50%)
        max_len        : số điều kiện tối đa trong vế trái của một luật
        prune          : bật cắt tỉa theo lỗi bi quan (tạo prCARs như bài báo)
        cf             : mức tin cậy của lỗi bi quan (C4.5 mặc định 25%)
        pos_label      : nhãn của lớp dương, dùng cho điểm số
        class_minsup   : True thì dùng minsup riêng theo lớp (Liu, Ma & Wong, 2000), xem class_min_counts
    """

    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5, max_len: int = 3,
                 prune: bool = True, cf: float = 0.25, pos_label: int = 1, class_minsup: bool = False):
        self.class_minsup = class_minsup
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.prune = prune
        self.cf = cf
        self.pos_label = pos_label

    # ---------------- CBA-RG ----------------
    def _generate_cars(self, table: CondsetTable, min_count) -> pd.DataFrame:
        cond_sup = table.counts.sum(axis=1)
        conf_all = table.counts / np.maximum(cond_sup[:, None], 1)
        pr_class = conf_all.argmax(axis=1)            # hòa thì lấy lớp đứng trước (bài báo chọn ngẫu nhiên)
        rule_sup = table.counts[np.arange(len(pr_class)), pr_class]
        conf = rule_sup / np.maximum(cond_sup, 1)
        # Bài báo viết "lớn hơn minconf"; ở đây dùng >= (chỉ khác khi confidence bằng đúng ngưỡng)
        thr = np.broadcast_to(np.asarray(min_count, dtype=float), (table.counts.shape[1],))[pr_class]
        keep = (rule_sup >= thr) & (conf >= self.min_confidence)
        idx = np.where(keep)[0]
        cars = pd.DataFrame({
            "cid": idx,
            "length": [len(table.condsets[i]) for i in idx],
            "cls_idx": pr_class[idx],
            "condsup": cond_sup[idx],
            "rulesup": rule_sup[idx],
            "confidence": conf[idx],
            "gen_order": table.gen_order[idx],
        })
        cars["support"] = cars["rulesup"] / table.n
        return cars

    def _prune_pessimistic(self, cars: pd.DataFrame, table: CondsetTable) -> pd.DataFrame:
        if cars.empty:
            return cars
        err_r = pessimistic_error(cars["condsup"] - cars["rulesup"], cars["condsup"], self.cf)
        # Gom mọi luật r- (bớt một điều kiện, cùng lớp) rồi tính lỗi bi quan một lần cho nhanh
        owner, sub_err, sub_tot = [], [], []
        for j, (cid, c) in enumerate(zip(cars["cid"], cars["cls_idx"])):
            cs = table.condsets[cid]
            for drop in range(len(cs)):
                sc = table.counts_of(cs[:drop] + cs[drop + 1:])
                if sc is None:        # không thể xảy ra với luật phổ biến, giữ để an toàn
                    continue
                owner.append(j)
                sub_err.append(sc.sum() - sc[c])
                sub_tot.append(sc.sum())
        keep = np.ones(len(cars), dtype=bool)
        if owner:
            owner = np.asarray(owner)
            e_sub = pessimistic_error(np.asarray(sub_err), np.asarray(sub_tot), self.cf)
            worse = np.asarray(err_r)[owner] > e_sub      # r tệ hơn ít nhất một r- thì bị cắt
            keep[np.unique(owner[worse])] = False
        return cars[keep].reset_index(drop=True)

    # ---------------- CBA-CB (M1) ----------------
    def _item_mask(self, a: int, v: int) -> np.ndarray:
        key = (a, v)
        m = self._mask_cache.get(key)
        if m is None:
            m = self._codes_fit[:, a] == v
            self._mask_cache[key] = m
        return m

    def _build_classifier(self, cars: pd.DataFrame, table: CondsetTable, y_idx: np.ndarray) -> None:
        order = cars.sort_values(["confidence", "rulesup", "gen_order"],
                                 ascending=[False, False, True]).reset_index(drop=True)
        n_cls = len(self.classes_)
        remaining = np.ones(len(y_idx), dtype=bool)
        is_cls = [y_idx == c for c in range(n_cls)]
        rule_errors = 0
        selected = []
        for r in order.itertuples(index=False):
            if not remaining.any():
                break
            cov = remaining.copy()
            for a, v in table.condsets[r.cid]:
                cov &= self._item_mask(a, v)
            n_cov = int(np.count_nonzero(cov))
            if n_cov == 0:
                continue
            n_correct = int(np.count_nonzero(cov & is_cls[r.cls_idx]))
            if n_correct == 0:
                continue
            rule_errors += n_cov - n_correct
            remaining &= ~cov
            rem_counts = np.bincount(y_idx[remaining], minlength=n_cls)
            default_idx = int(rem_counts.argmax()) if rem_counts.sum() > 0 else int(table.class_totals.argmax())
            default_err = int(rem_counts.sum() - rem_counts[default_idx])
            selected.append({"cid": r.cid, "cls_idx": r.cls_idx, "confidence": r.confidence,
                             "support": r.support, "rulesup": r.rulesup, "condsup": r.condsup,
                             "train_covered": n_cov, "train_correct": n_correct,
                             "default_idx": default_idx, "rem_counts": rem_counts,
                             "total_errors": rule_errors + default_err})
        self.selection_trace_ = pd.DataFrame(
            [{k: v for k, v in s.items() if k != "rem_counts"} for s in selected])
        pos_idx = int(np.where(self.classes_ == self.pos_label)[0][0]) if self.pos_label in self.classes_ else 0
        if not selected:
            self.rules_ = []
            self.default_idx_ = int(table.class_totals.argmax())
            self.default_score_ = table.class_totals[pos_idx] / table.class_totals.sum()
            return
        totals = np.array([s["total_errors"] for s in selected])
        cut = int(np.argmin(totals))                   # luật đầu tiên có tổng lỗi nhỏ nhất
        self.rules_ = selected[:cut + 1]
        self.default_idx_ = selected[cut]["default_idx"]
        rc = selected[cut]["rem_counts"]
        self.default_score_ = (rc[pos_idx] / rc.sum()) if rc.sum() > 0 else \
            table.class_totals[pos_idx] / table.class_totals.sum()
        # Điểm số của từng luật cho lớp dương: tỷ lệ lớp dương trong mọi dòng khớp condset
        for s in self.rules_:
            cnt = table.counts[s["cid"]]
            s["pos_score"] = cnt[pos_idx] / cnt.sum()
            s["items"] = table.condsets[s["cid"]]

    # ---------------- Giao diện chính ----------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CBAClassifier":
        t0 = time.perf_counter()
        codes, self.feature_names_, self.cards_ = encode_codes(X)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        n = len(y_idx)
        class_totals = np.bincount(y_idx, minlength=len(self.classes_))
        min_count = class_min_counts(self.min_support, class_totals, n, self.class_minsup)
        table = count_condsets(codes, self.cards_, y_idx, len(self.classes_), min_count, self.max_len)
        cars = self._generate_cars(table, min_count)
        self.n_condsets_ = len(table.condsets)
        self.n_cars_ = len(cars)
        t1 = time.perf_counter()
        if self.prune:
            cars = self._prune_pessimistic(cars, table)
        self.n_rules_after_prune_ = len(cars)
        t2 = time.perf_counter()
        self._codes_fit = codes
        self._mask_cache: dict = {}
        self._build_classifier(cars, table, y_idx)
        t3 = time.perf_counter()
        del self._codes_fit, self._mask_cache
        self.timing_ = {"rule_generation_s": t1 - t0, "pruning_s": t2 - t1, "classifier_building_s": t3 - t2}
        self.cars_ = cars
        self.table_ = table
        return self

    def _apply(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        codes = X[self.feature_names_].to_numpy(dtype=np.int64) + 1
        n = len(codes)
        pred = np.full(n, self.default_idx_, dtype=np.int64)
        score = np.full(n, self.default_score_, dtype=float)
        fired = np.full(n, -1, dtype=np.int64)          # chỉ số luật quyết định; -1 là lớp mặc định
        free = np.ones(n, dtype=bool)
        for k, s in enumerate(self.rules_):
            m = free.copy()
            for a, v in s["items"]:
                m &= codes[:, a] == v
            pred[m] = s["cls_idx"]
            score[m] = s["pos_score"]
            fired[m] = k
            free &= ~m
        return self.classes_[pred], score, fired

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._apply(X)[0]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return self._apply(X)[1]

    def fired_rule(self, X: pd.DataFrame) -> np.ndarray:
        return self._apply(X)[2]

    def describe_condset(self, items: Condset, labelers: dict | None = None) -> str:
        parts = []
        for a, v in items:
            name = self.feature_names_[a]
            code = v - 1
            lab = labelers[name].get(code, str(code)) if labelers and name in labelers else str(code)
            parts.append(f"{name} {lab}" if lab[:1] in "<>[" else f"{name} = {lab}")
        return " VÀ ".join(parts)

    def rules_table(self, labelers: dict | None = None) -> pd.DataFrame:
        """Bảng luật của bộ phân lớp theo đúng thứ tự áp dụng, kèm lớp mặc định ở dòng cuối."""
        rows = []
        for k, s in enumerate(self.rules_, start=1):
            rows.append({"thu_tu": k, "dieu_kien": self.describe_condset(s["items"], labelers),
                         "lop": self.classes_[s["cls_idx"]], "support": s["support"],
                         "confidence": s["confidence"], "so_dong_phu_khi_chon": s["train_covered"],
                         "so_dong_dung_khi_chon": s["train_correct"], "tong_loi_tich_luy": s["total_errors"]})
        rows.append({"thu_tu": len(rows) + 1, "dieu_kien": "(lớp mặc định)", "lop": self.classes_[self.default_idx_],
                     "support": np.nan, "confidence": np.nan, "so_dong_phu_khi_chon": np.nan,
                     "so_dong_dung_khi_chon": np.nan, "tong_loi_tich_luy": np.nan})
        return pd.DataFrame(rows)
