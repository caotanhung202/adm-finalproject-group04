"""Đọc dữ liệu bàn giao từ bước 2, tạo bảng cặp (người dùng, sản phẩm), gắn nhãn và chia tập theo người dùng.

Định nghĩa bài toán (task T34):
    - Đối tượng: mỗi cặp (user_id, product_id) mà người dùng đã mua ít nhất một lần trong các đơn prior.
    - Chỉ lấy người dùng có đơn thuộc eval_set = train (người dùng thuộc test không có nhãn công bố).
    - Nhãn = 1 nếu sản phẩm xuất hiện trong đơn train của người dùng đó, ngược lại = 0.

Chia tập (task T38):
    - Chia theo user_id, không chia theo dòng, để mọi cặp của một người dùng nằm trọn trong một tập.
    - Ba phần: "fit" (huấn luyện), "valid" (kiểm định, dùng chọn siêu tham số và ngưỡng),
      "holdout" (kiểm tra, chỉ dùng một lần để báo cáo kết quả cuối).
    - Tên "fit/valid/holdout" được chọn có chủ ý để không nhầm với eval_set train/test của Instacart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Tên tệp do bước 2 (Thiên Trang) bàn giao
B2_FILES = {
    "orders_prior": "orders_prior_sample.parquet",
    "orders_train": "orders_train_sample.parquet",
    "op_prior": "order_products_prior_sample.parquet",
    "op_train": "order_products_train_sample.parquet",
    "products": "products_full.parquet",
}

ORDER_COLS = ["order_id", "user_id", "order_number", "order_dow",
              "order_hour_of_day", "days_since_prior_order"]
OP_COLS = ["order_id", "product_id", "add_to_cart_order", "reordered"]


@dataclass
class B2Data:
    """Năm bảng đầu vào của bước 5."""

    orders_prior: pd.DataFrame
    orders_train: pd.DataFrame
    op_prior: pd.DataFrame
    op_train: pd.DataFrame
    products: pd.DataFrame


def load_b2_data(data_dir: str | Path) -> B2Data:
    """Đọc năm tệp Parquet của bước 2 và ép kiểu gọn để tiết kiệm bộ nhớ trên Colab."""
    data_dir = Path(data_dir)
    missing = [f for f in B2_FILES.values() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu tệp trong {data_dir}: {missing}")

    orders_prior = pd.read_parquet(data_dir / B2_FILES["orders_prior"], columns=ORDER_COLS)
    orders_train = pd.read_parquet(data_dir / B2_FILES["orders_train"], columns=ORDER_COLS)
    op_prior = pd.read_parquet(data_dir / B2_FILES["op_prior"], columns=OP_COLS)
    op_train = pd.read_parquet(data_dir / B2_FILES["op_train"], columns=OP_COLS)
    products = pd.read_parquet(data_dir / B2_FILES["products"])

    for df in (orders_prior, orders_train):
        df["order_id"] = df["order_id"].astype("int64")
        df["user_id"] = df["user_id"].astype("int64")
        df["order_number"] = df["order_number"].astype("int16")
        df["order_dow"] = df["order_dow"].astype("int8")
        df["order_hour_of_day"] = df["order_hour_of_day"].astype("int8")
        df["days_since_prior_order"] = df["days_since_prior_order"].astype("float32")
    for df in (op_prior, op_train):
        df["order_id"] = df["order_id"].astype("int64")
        df["product_id"] = df["product_id"].astype("int32")
        df["add_to_cart_order"] = df["add_to_cart_order"].astype("int16")
        df["reordered"] = df["reordered"].astype("int8")
    products["product_id"] = products["product_id"].astype("int32")
    return B2Data(orders_prior, orders_train, op_prior, op_train, products)


def integrity_report(d: B2Data) -> pd.DataFrame:
    """Các kiểm tra đầu vào trước khi tạo nhãn. Mọi dòng 'số vi phạm' phải bằng 0.

    Riêng dòng 'đơn prior không có dòng sản phẩm' dùng để phát hiện trường hợp bảng
    order_products_prior_sample bị cắt bớt (số dòng 2.066.000 là một số tròn, cần xác nhận).
    """
    users_prior = set(d.orders_prior["user_id"].unique())
    users_train = set(d.orders_train["user_id"].unique())
    prior_ids = set(d.orders_prior["order_id"])
    train_ids = set(d.orders_train["order_id"])
    op_prior_ids = set(d.op_prior["order_id"].unique())
    op_train_ids = set(d.op_train["order_id"].unique())

    rows = [
        ("Số người dùng có đơn prior", len(users_prior), None),
        ("Số người dùng có đơn train", len(users_train), None),
        ("Người dùng có đơn train nhưng không có lịch sử prior", len(users_train - users_prior), 0),
        ("Người dùng có nhiều hơn một đơn train",
         int((d.orders_train.groupby("user_id").size() > 1).sum()), 0),
        ("Dòng sản phẩm prior trỏ tới đơn không có trong orders_prior", len(op_prior_ids - prior_ids), 0),
        ("Dòng sản phẩm train trỏ tới đơn không có trong orders_train", len(op_train_ids - train_ids), 0),
        ("Đơn prior không có dòng sản phẩm nào", len(prior_ids - op_prior_ids), 0),
        ("Đơn train không có dòng sản phẩm nào", len(train_ids - op_train_ids), 0),
        ("Cặp (order_id, product_id) trùng trong prior",
         int(d.op_prior.duplicated(["order_id", "product_id"]).sum()), 0),
        ("Cặp (order_id, product_id) trùng trong train",
         int(d.op_train.duplicated(["order_id", "product_id"]).sum()), 0),
        ("product_id không có trong products_full",
         int((~d.op_prior["product_id"].isin(d.products["product_id"])).sum()
             + (~d.op_train["product_id"].isin(d.products["product_id"])).sum()), 0),
    ]
    out = pd.DataFrame(rows, columns=["Kiểm tra", "Giá trị", "Kỳ vọng"])
    out["Đạt"] = np.where(out["Kỳ vọng"].isna(), "", np.where(out["Giá trị"] == out["Kỳ vọng"], "Đạt", "KHÔNG ĐẠT"))
    return out


def build_labels(d: B2Data) -> tuple[pd.DataFrame, dict]:
    """Tạo bảng cặp (user_id, product_id, label).

    Trả về:
        pairs: mỗi dòng là một cặp người dùng đã từng mua sản phẩm trong prior, kèm nhãn 0/1.
        stats: các con số để báo cáo và để kiểm tra chéo.

    Kiểm tra chéo quan trọng: trong đơn train, cột reordered = 1 nghĩa là người dùng đã mua sản phẩm
    trước đó. Vì vậy số cặp có nhãn 1 phải bằng số dòng train có reordered = 1 (nếu dữ liệu nhất quán).
    """
    users_train = d.orders_train["user_id"].unique()

    # Lịch sử prior của những người dùng có đơn train
    prior = d.op_prior.merge(d.orders_prior[["order_id", "user_id"]], on="order_id", how="inner")
    prior = prior[prior["user_id"].isin(users_train)]
    pairs = prior[["user_id", "product_id"]].drop_duplicates().reset_index(drop=True)

    # Sản phẩm trong đơn train của từng người dùng
    train = d.op_train.merge(d.orders_train[["order_id", "user_id"]], on="order_id", how="inner")
    train_pairs = train[["user_id", "product_id", "reordered"]]

    pairs = pairs.merge(train_pairs[["user_id", "product_id"]].assign(label=1),
                        on=["user_id", "product_id"], how="left")
    pairs["label"] = pairs["label"].fillna(0).astype("int8")

    n_train_lines = len(train_pairs)
    n_train_reorder = int(train_pairs["reordered"].sum())
    n_pos = int(pairs["label"].sum())
    stats = {
        "so_nguoi_dung": int(pairs["user_id"].nunique()),
        "so_cap": int(len(pairs)),
        "so_cap_nhan_1": n_pos,
        "so_cap_nhan_0": int(len(pairs) - n_pos),
        "ty_le_nhan_1": n_pos / len(pairs),
        "ty_le_0_tren_1": (len(pairs) - n_pos) / max(n_pos, 1),
        "so_cap_trung_binh_moi_nguoi": len(pairs) / pairs["user_id"].nunique(),
        "so_dong_san_pham_trong_don_train": n_train_lines,
        "so_dong_train_reordered_1": n_train_reorder,
        # Sản phẩm mới (chưa từng mua) trong đơn train nằm ngoài tập ứng viên:
        # đây là giới hạn trên của recall nếu xét theo toàn bộ đơn train.
        "ty_le_dong_train_la_san_pham_moi": 1 - n_train_reorder / n_train_lines,
        "kiem_tra_nhan_1_bang_reordered_train": n_pos == n_train_reorder,
    }
    return pairs, stats


def split_users(user_ids: np.ndarray, holdout_frac: float = 0.20, valid_frac: float = 0.10,
                seed: int = 42) -> pd.DataFrame:
    """Chia người dùng thành fit / valid / holdout bằng hoán vị ngẫu nhiên có seed cố định.

    valid_frac và holdout_frac tính trên tổng số người dùng (mặc định 70% / 10% / 20%).
    """
    users = np.sort(np.unique(user_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(users)
    n = len(perm)
    n_hold = int(round(n * holdout_frac))
    n_valid = int(round(n * valid_frac))
    split = np.empty(n, dtype=object)
    split[:n_hold] = "holdout"
    split[n_hold:n_hold + n_valid] = "valid"
    split[n_hold + n_valid:] = "fit"
    out = pd.DataFrame({"user_id": perm, "split": split}).sort_values("user_id").reset_index(drop=True)
    return out


def split_summary(pairs: pd.DataFrame, user_split: pd.DataFrame) -> pd.DataFrame:
    """Bảng số người dùng, số cặp và tỷ lệ nhãn 1 theo từng phần; dùng cho mục 3.6 và 4.6."""
    df = pairs.merge(user_split, on="user_id", how="left")
    if df["split"].isna().any():
        raise ValueError("Có người dùng chưa được gán vào phần nào.")
    g = df.groupby("split").agg(so_nguoi_dung=("user_id", "nunique"), so_cap=("label", "size"),
                                so_nhan_1=("label", "sum"))
    g["ty_le_nhan_1"] = g["so_nhan_1"] / g["so_cap"]
    tot = pd.DataFrame({"so_nguoi_dung": [df["user_id"].nunique()], "so_cap": [len(df)],
                        "so_nhan_1": [int(df["label"].sum())]}, index=["tong"])
    tot["ty_le_nhan_1"] = tot["so_nhan_1"] / tot["so_cap"]
    return pd.concat([g.reindex(["fit", "valid", "holdout"]), tot])


def assert_no_user_overlap(user_split: pd.DataFrame) -> None:
    """Một người dùng chỉ thuộc đúng một phần (điểm chống rò rỉ dữ liệu, task T45 sẽ kiểm tra lại)."""
    counts = user_split.groupby("user_id")["split"].nunique()
    if (counts > 1).any():
        raise AssertionError("Có người dùng nằm ở nhiều hơn một phần.")
