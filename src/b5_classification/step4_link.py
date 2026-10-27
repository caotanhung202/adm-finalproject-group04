"""Nối kết quả bước 4 (luật kết hợp giữa các gian hàng) sang bước 5 thành một thuộc tính.

Yêu cầu số 5 của đề: phân lớp "dựa trên kết quả bước 4". Ngoài việc CBA và CMAR bản thân đã là phân lớp
dựa trên luật kết hợp, module này đưa trực tiếp các luật mà bước 4 tìm được vào bài toán:

    r4_lift(u, p) = lift lớn nhất trong các luật bước 4 dạng A -> {gian hàng của p} sao cho mọi gian hàng
                    trong vế trái A đều có mặt trong đơn prior gần nhất của người dùng u.
                    Không có luật nào khớp thì r4_lift = 1 (trung tính, đúng nghĩa lift = 1).

Diễn giải: nếu giỏ gần nhất của người dùng chứa đúng "điều kiện" của một luật mạnh trỏ tới gian hàng của
sản phẩm, liệu sản phẩm đó có dễ được mua lại hơn không? Thực nghiệm so sánh CBA có và không có thuộc tính
này (ablation) để trả lời trực tiếp câu hỏi "luật bước 4 có giúp dự đoán không".

Chỉ dùng luật có đúng một gian hàng ở vế phải và lift > 1. Luật chỉ được dùng như một nguồn thông tin về
giỏ hàng prior, không dùng nhãn, nên không gây rò rỉ nhãn.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_ANT_CANDIDATES = ["antecedents", "antecedent", "lhs", "ve_trai", "X"]
_CON_CANDIDATES = ["consequents", "consequent", "rhs", "ve_phai", "Y"]


def _pick(cols: list[str], candidates: list[str]) -> str:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"Không tìm thấy cột nào trong {candidates}; các cột hiện có: {cols}")


def _tokens(cell) -> list[str]:
    """Tách một ô vế luật thành danh sách tên hoặc mã.

    Chấp nhận các dạng: "frozenset({'yogurt', 'milk'})" (mlxtend), "['yogurt', 'milk']",
    "yogurt + milk", "yogurt|milk", "24, 83".
    """
    if isinstance(cell, (list, tuple, set, frozenset)):
        return [str(t).strip() for t in cell]
    s = str(cell)
    s = re.sub(r"frozenset\(|[{}\[\]()'\"]", " ", s)
    parts = [p.strip() for p in re.split(r"\s*[,|+;]\s*", s)]
    return [p for p in parts if p]


def load_step4_rules(path: str, products: pd.DataFrame, level: str = "aisle",
                     antecedent_col: str | None = None, consequent_col: str | None = None) -> pd.DataFrame:
    """Đọc tệp CSV luật của bước 4 và chuẩn hóa về mã gian hàng (hoặc ngành hàng).

    Trả về DataFrame: antecedent (tuple mã), consequent (mã), support, confidence, lift.
    """
    key, name = ("aisle_id", "aisle") if level == "aisle" else ("department_id", "department")
    raw = pd.read_csv(path, encoding="utf-8-sig")   # tệp của B4 có ký tự BOM ở đầu
    a_col = antecedent_col or _pick(list(raw.columns), _ANT_CANDIDATES)
    c_col = consequent_col or _pick(list(raw.columns), _CON_CANDIDATES)

    name_to_id = {}
    if name in products.columns:
        cat = products[[key, name]].drop_duplicates()
        name_to_id = {str(n).strip().lower(): int(i) for i, n in zip(cat[key], cat[name])}

    def to_id(tok: str) -> int:
        if re.fullmatch(r"\d+", tok):
            return int(tok)
        m = re.fullmatch(r"(?:aisle|department|dept)_(\d+)", tok.strip().lower())
        if m:                                    # dạng "aisle_24" mà B4 xuất ra
            return int(m.group(1))
        t = tok.strip().lower()
        if t not in name_to_id:
            raise ValueError(f"Không nhận ra '{tok}' là {name} nào. Kiểm tra cấp luật (level) hoặc tên cột.")
        return name_to_id[t]

    rows = []
    for _, r in raw.iterrows():
        ant = tuple(sorted(to_id(t) for t in _tokens(r[a_col])))
        con = [to_id(t) for t in _tokens(r[c_col])]
        if len(con) != 1:
            continue
        rows.append({"antecedent": ant, "consequent": con[0],
                     "support": float(r.get("support", np.nan)),
                     "confidence": float(r.get("confidence", np.nan)),
                     "lift": float(r["lift"])})
    out = pd.DataFrame(rows)
    return out[out["lift"] > 1].reset_index(drop=True)


def mine_aisle_rules(baskets: pd.DataFrame, key: str = "aisle_id", min_support: float = 0.02,
                     min_confidence: float = 0.5, max_antecedent: int = 2) -> pd.DataFrame:
    """Phương án dự phòng: tự khai thác luật gian hàng khi chưa có CSV của bước 4.

    baskets: bảng (order_id, key) đã bỏ trùng. Chỉ nên truyền đơn prior của người dùng fit.
    Đếm hỗ trợ bằng phép nhân ma trận nhị phân (cặp) và phép AND hai cột rồi nhân (bộ ba), cho kết quả
    giống Apriori với độ dài tập mục tối đa max_antecedent + 1.
    """
    orders = baskets["order_id"].astype("category")
    items = baskets[key].astype("category")
    n_orders = len(orders.cat.categories)
    ids = np.asarray(items.cat.categories)
    X = np.zeros((n_orders, len(ids)), dtype=np.float32)
    X[orders.cat.codes.to_numpy(), items.cat.codes.to_numpy()] = 1.0
    n = float(n_orders)
    single = X.sum(axis=0)
    pair = X.T @ X
    min_count = min_support * n
    rows = []

    def add(ant_idx: tuple, c: int, both: float, ant_count: float) -> None:
        conf = both / ant_count
        lift = conf / (single[c] / n)
        if conf >= min_confidence and lift > 1:
            rows.append({"antecedent": tuple(sorted(int(ids[i]) for i in ant_idx)), "consequent": int(ids[c]),
                         "support": both / n, "confidence": conf, "lift": lift})

    freq_pairs = []
    for a in range(len(ids)):
        if single[a] < min_count:
            continue
        for c in range(len(ids)):
            if c != a and pair[a, c] >= min_count:
                add((a,), c, pair[a, c], single[a])
                if a < c:
                    freq_pairs.append((a, c))
    if max_antecedent >= 2:
        for a, b in freq_pairs:
            v = X[:, a] * X[:, b]
            tri = v @ X
            for c in np.where(tri >= min_count)[0]:
                if c not in (a, b):
                    add((a, b), int(c), float(tri[c]), float(pair[a, b]))
    return pd.DataFrame(rows)


def r4_lift_feature(pairs: pd.DataFrame, orders_prior: pd.DataFrame, op_prior: pd.DataFrame,
                    products: pd.DataFrame, rules: pd.DataFrame, level: str = "aisle") -> pd.Series:
    """Tính r4_lift cho từng cặp trong pairs (giữ đúng thứ tự dòng)."""
    key = "aisle_id" if level == "aisle" else "department_id"
    users = pairs["user_id"].unique()
    o = orders_prior[orders_prior["user_id"].isin(users)]
    last = o.loc[o.groupby("user_id")["order_number"].idxmax(), ["user_id", "order_id"]]
    ctx = (op_prior[op_prior["order_id"].isin(last["order_id"])]
           .merge(last, on="order_id").merge(products[["product_id", key]], on="product_id")
           [["user_id", key]].drop_duplicates())

    user_index = {u: i for i, u in enumerate(users)}
    cat_ids = np.sort(products[key].unique())
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    M = np.zeros((len(users), len(cat_ids)), dtype=bool)
    M[ctx["user_id"].map(user_index).to_numpy(), ctx[key].map(cat_index).to_numpy()] = True

    best = np.ones((len(users), len(cat_ids)), dtype=np.float32)
    for _, r in rules.iterrows():
        cols = [cat_index[a] for a in r["antecedent"] if a in cat_index]
        if len(cols) != len(r["antecedent"]) or r["consequent"] not in cat_index:
            continue
        match = M[:, cols].all(axis=1)
        j = cat_index[r["consequent"]]
        best[match, j] = np.maximum(best[match, j], r["lift"])

    p_cat = pairs[["product_id"]].merge(products[["product_id", key]], on="product_id", how="left")[key]
    ui = pairs["user_id"].map(user_index).to_numpy()
    ci = p_cat.map(cat_index).to_numpy()
    return pd.Series(best[ui, ci], index=pairs.index, name="r4_lift")
