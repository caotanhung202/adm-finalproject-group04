"""Sinh dữ liệu GIẢ LẬP có cùng lược đồ với năm tệp Parquet của bước 2.

Chỉ dùng để kiểm tra mã chạy không lỗi. Không một con số nào từ dữ liệu này được đưa vào báo cáo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(out_dir: str, n_users: int = 800, n_products: int = 400, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    aisles = rng.integers(1, 31, n_products)
    products = pd.DataFrame({
        "product_id": np.arange(1, n_products + 1, dtype=np.int64),
        "product_name": [f"Product {i}" for i in range(1, n_products + 1)],
        "aisle_id": aisles,
        "aisle": [f"aisle {a}" for a in aisles],
        "department_id": (aisles - 1) // 6 + 1,
    })
    products["department"] = [f"dept {d}" for d in products["department_id"]]
    popularity = rng.zipf(1.6, n_products).astype(float)
    popularity /= popularity.sum()

    orders, lines = [], []
    oid = 1
    for u in range(1, n_users + 1):
        n_ord = int(rng.integers(4, 40))
        fav = rng.choice(n_products, size=int(rng.integers(5, 25)), replace=False, p=popularity) + 1
        seen: set = set()
        for k in range(1, n_ord + 1):
            gap = np.nan if k == 1 else float(min(30, rng.geometric(1 / 8)))
            eval_set = "train" if k == n_ord else "prior"
            orders.append((oid, u, eval_set, k, int(rng.integers(0, 7)), int(rng.integers(6, 23)), gap))
            size = int(rng.integers(3, 15))
            basket = set(rng.choice(fav, size=min(size, len(fav)), replace=False))
            basket |= set(rng.choice(n_products, size=int(rng.integers(0, 4)), p=popularity) + 1)
            for pos, p in enumerate(basket, start=1):
                lines.append((oid, int(p), pos, int(p in seen), eval_set))
            seen |= basket
            oid += 1
    o = pd.DataFrame(orders, columns=["order_id", "user_id", "eval_set", "order_number", "order_dow",
                                      "order_hour_of_day", "days_since_prior_order"])
    li = pd.DataFrame(lines, columns=["order_id", "product_id", "add_to_cart_order", "reordered", "eval_set"])
    o[o.eval_set == "prior"].to_parquet(out / "orders_prior_sample.parquet", index=False)
    o[o.eval_set == "train"].to_parquet(out / "orders_train_sample.parquet", index=False)
    li[li.eval_set == "prior"].drop(columns="eval_set").to_parquet(out / "order_products_prior_sample.parquet", index=False)
    li[li.eval_set == "train"].drop(columns="eval_set").to_parquet(out / "order_products_train_sample.parquet", index=False)
    products.to_parquet(out / "products_full.parquet", index=False)

    # Tệp luật giả lập theo định dạng mlxtend để thử bộ đọc luật bước 4
    pd.DataFrame({
        "antecedents": ["frozenset({'aisle 1'})", "frozenset({'aisle 2', 'aisle 3'})"],
        "consequents": ["frozenset({'aisle 4'})", "frozenset({'aisle 1'})"],
        "support": [0.1, 0.05], "confidence": [0.6, 0.55], "lift": [1.4, 1.2],
    }).to_csv(out / "rules_aisle_step4.csv", index=False)


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
