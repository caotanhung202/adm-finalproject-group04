"""Rời rạc hóa có giám sát theo entropy với tiêu chuẩn dừng MDL (Fayyad và Irani, 1993).

Bài CBA (Liu, Hsu và Ma, 1998) rời rạc hóa thuộc tính liên tục bằng chính phương pháp này trước khi
khai thác luật lớp. Cài đặt ở đây:

    1. Với một đoạn S, thử mọi điểm cắt T nằm giữa hai giá trị liên tiếp khác nhau, chọn T làm cực tiểu
       entropy có trọng số E(T; S) = |S1|/|S| * Ent(S1) + |S2|/|S| * Ent(S2).
    2. Chấp nhận T khi Gain(T; S) > [log2(N - 1) + Delta(T; S)] / N, trong đó
       Delta(T; S) = log2(3^k - 2) - [k * Ent(S) - k1 * Ent(S1) - k2 * Ent(S2)],
       k, k1, k2 là số lớp có mặt trong S, S1, S2; N = |S|.
    3. Lặp đệ quy trên hai đoạn con.

Điểm khác với bản gốc (ghi rõ trong báo cáo): thêm tham số max_intervals. Với hàng trăm nghìn dòng,
ngưỡng MDL rất nhỏ (log2 N / N) nên thuật toán gốc chấp nhận rất nhiều điểm cắt (bước 2 cũ ra 13 khoảng).
Mỗi khoảng là một item của CBA, nhiều item làm số luật tăng theo cấp số nhân. Vì vậy chỉ giữ tối đa
max_intervals khoảng bằng cách tách theo thứ tự "đoạn nào có mức giảm entropy tuyệt đối lớn nhất thì tách
trước" (best-first); mọi điểm cắt được giữ vẫn phải qua tiêu chuẩn MDL.

Giá trị thiếu (NaN) được đưa vào một item riêng (mã -1) thay vì bị bỏ đi.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _entropy_rows(counts: np.ndarray) -> np.ndarray:
    """Entropy (cơ số 2) cho từng hàng của ma trận đếm lớp."""
    tot = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, counts / tot, 0.0)
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=1)


def _best_cut(cum: np.ndarray, lo: int, hi: int):
    """Tìm điểm cắt tốt nhất trong đoạn giá trị duy nhất [lo, hi) và kiểm tra tiêu chuẩn MDL.

    cum là ma trận đếm lớp cộng dồn theo giá trị duy nhất đã sắp xếp, có thêm hàng 0 ở đầu.
    Trả về (gain_tuyet_doi, chi_so_cat) nếu được chấp nhận, ngược lại None.
    """
    if hi - lo < 2:
        return None
    total = cum[hi] - cum[lo]
    n = total.sum()
    if n < 2:
        return None
    # Mọi vị trí cắt i trong (lo, hi): trái = [lo, i), phải = [i, hi)
    left = cum[lo + 1:hi] - cum[lo]
    right = total - left
    nl = left.sum(axis=1)
    nr = right.sum(axis=1)
    ent_l = _entropy_rows(left)
    ent_r = _entropy_rows(right)
    e = (nl * ent_l + nr * ent_r) / n
    j = int(np.argmin(e))
    ent_s = _entropy_rows(total[None, :])[0]
    gain = ent_s - e[j]
    k = int((total > 0).sum())
    k1 = int((left[j] > 0).sum())
    k2 = int((right[j] > 0).sum())
    delta = np.log2(3 ** k - 2) - (k * ent_s - k1 * ent_l[j] - k2 * ent_r[j])
    threshold = (np.log2(n - 1) + delta) / n
    if gain > threshold:
        return gain * n, lo + 1 + j
    return None


def mdlp_cut_points(x: np.ndarray, y: np.ndarray, max_intervals: int | None = 5) -> list[float]:
    """Trả về danh sách điểm cắt (tăng dần) cho một thuộc tính số x theo nhãn y."""
    mask = ~np.isnan(x)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return []
    classes, y_idx = np.unique(y, return_inverse=True)
    uniq, inv = np.unique(x, return_inverse=True)
    counts = np.zeros((len(uniq), len(classes)), dtype=np.int64)
    np.add.at(counts, (inv, y_idx), 1)
    cum = np.vstack([np.zeros((1, len(classes)), dtype=np.int64), np.cumsum(counts, axis=0)])

    limit = max_intervals if max_intervals else np.inf
    cuts_idx: list[int] = []
    heap: list = []
    first = _best_cut(cum, 0, len(uniq))
    if first:
        heapq.heappush(heap, (-first[0], 0, len(uniq), first[1]))
    while heap and len(cuts_idx) + 1 < limit:
        _, lo, hi, i = heapq.heappop(heap)
        cuts_idx.append(i)
        for a, b in ((lo, i), (i, hi)):
            res = _best_cut(cum, a, b)
            if res:
                heapq.heappush(heap, (-res[0], a, b, res[1]))
    # Điểm cắt đặt ở trung điểm giữa hai giá trị duy nhất liền kề
    return sorted(float((uniq[i - 1] + uniq[i]) / 2) for i in cuts_idx)


@dataclass
class Discretizer:
    """Học điểm cắt trên phần fit, áp dụng cho mọi phần. Lưu được ra JSON để bước 2 trích dẫn."""

    max_intervals: int | None = 5
    cuts: dict[str, list[float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, features: list[str], y: np.ndarray) -> "Discretizer":
        for f in features:
            self.cuts[f] = mdlp_cut_points(df[f].to_numpy(dtype=float), np.asarray(y), self.max_intervals)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mã khoảng 0..k (theo thứ tự tăng dần); NaN nhận mã -1."""
        out = pd.DataFrame(index=df.index)
        for f, c in self.cuts.items():
            v = df[f].to_numpy(dtype=float)
            code = np.searchsorted(np.asarray(c, dtype=float), v, side="right").astype("int16")
            code[np.isnan(v)] = -1
            out[f] = code
        return out

    def interval_labels(self, f: str, decimals: int = 3) -> dict[int, str]:
        """Nhãn dễ đọc cho từng mã khoảng, ví dụ {0: '< 2,5', 1: '[2,5; 7,5)', 2: '>= 7,5'}."""
        c = self.cuts[f]

        def fmt(v: float) -> str:
            s = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
            return s.replace(".", ",")

        labels = {-1: "thiếu"}
        if not c:
            labels[0] = "mọi giá trị"
            return labels
        labels[0] = f"< {fmt(c[0])}"
        for i in range(1, len(c)):
            labels[i] = f"[{fmt(c[i - 1])}; {fmt(c[i])})"
        labels[len(c)] = f">= {fmt(c[-1])}"
        return labels

    def summary(self, df_fit: pd.DataFrame, y_fit: np.ndarray) -> pd.DataFrame:
        """Bảng khoảng, số cặp và tỷ lệ nhãn 1 trong từng khoảng (trên phần fit)."""
        codes = self.transform(df_fit)
        rows = []
        for f in self.cuts:
            lab = self.interval_labels(f)
            g = pd.DataFrame({"code": codes[f].to_numpy(), "y": y_fit}).groupby("code")["y"].agg(["size", "mean"])
            for code, r in g.iterrows():
                rows.append({"thuoc_tinh": f, "ma": int(code), "khoang": lab.get(int(code), str(code)),
                             "so_cap": int(r["size"]), "ty_le_nhan_1": float(r["mean"])})
        return pd.DataFrame(rows)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"max_intervals": self.max_intervals, "cuts": self.cuts}, fh, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Discretizer":
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        return cls(max_intervals=obj["max_intervals"], cuts=obj["cuts"])
