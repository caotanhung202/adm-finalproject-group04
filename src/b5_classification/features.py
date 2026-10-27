"""Tạo thuộc tính cho từng cặp (người dùng, sản phẩm) (task T37).

Nguyên tắc chống rò rỉ dữ liệu:
    1. Mọi thuộc tính hành vi chỉ lấy từ các đơn prior. Nội dung đơn train (sản phẩm nào, thứ tự thêm
       vào giỏ, cột reordered của đơn train) chỉ dùng để tạo nhãn, không bao giờ là thuộc tính.
    2. Thông tin duy nhất lấy từ đơn train là days_since_prior_order của chính đơn đó (khoảng cách từ
       đơn prior cuối tới đơn cần dự đoán). Thông tin này đã biết khi khách bắt đầu đặt đơn.
    3. Thuộc tính tổng hợp theo sản phẩm, gian hàng, ngành hàng (tỷ lệ mua lại) chỉ tính trên người dùng
       thuộc phần "fit", rồi gán sang valid và holdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Danh sách thuộc tính cuối cùng; giữ ngắn vì CBA bùng nổ số luật khi có nhiều thuộc tính
FEATURE_DESCRIPTIONS = {
    "up_n_bought": "Số đơn prior mà người dùng đã mua sản phẩm",
    "up_rate": "Tỷ lệ đơn prior của người dùng có chứa sản phẩm (up_n_bought / u_n_orders)",
    "up_orders_since_last": "Số đơn prior đã trôi qua kể từ lần cuối mua sản phẩm",
    "up_days_since_last": "Số ngày ước tính từ lần cuối mua sản phẩm tới đơn cần dự đoán",
    "up_mean_cart_pos": "Vị trí trung bình khi thêm sản phẩm vào giỏ",
    "u_n_orders": "Số đơn prior của người dùng",
    "u_mean_basket": "Số sản phẩm trung bình mỗi đơn của người dùng",
    "u_mean_gap": "Khoảng cách trung bình (ngày) giữa hai đơn liên tiếp của người dùng",
    "u_reorder_ratio": "Tỷ lệ dòng mua lại của người dùng (không tính đơn đầu)",
    "u_days_to_target": "Số ngày từ đơn prior cuối tới đơn cần dự đoán (days_since_prior_order của đơn train)",
    "p_reorder_rate": "Tỷ lệ mua lại của sản phẩm (làm trơn về tỷ lệ của gian hàng), tính trên người dùng fit",
    "a_reorder_rate": "Tỷ lệ mua lại của gian hàng, tính trên người dùng fit",
    "d_reorder_rate": "Tỷ lệ mua lại của ngành hàng, tính trên người dùng fit",
}


def _prior_lines(orders_prior: pd.DataFrame, op_prior: pd.DataFrame, users: np.ndarray) -> pd.DataFrame:
    """Ghép dòng sản phẩm prior với thông tin đơn và ngày cộng dồn của từng người dùng."""
    o = orders_prior[orders_prior["user_id"].isin(users)].sort_values(["user_id", "order_number"]).copy()
    # Ngày cộng dồn: đơn đầu có days_since_prior_order trống nên coi là ngày 0 của người dùng.
    # Lưu ý: giá trị bị chặn trần ở 30 nên khoảng cách thật có thể dài hơn (sai số đã biết).
    o["cum_days"] = o.groupby("user_id")["days_since_prior_order"].transform(
        lambda s: s.fillna(0).cumsum()).astype("float32")
    lines = op_prior.merge(o[["order_id", "user_id", "order_number", "cum_days"]], on="order_id", how="inner")
    return lines, o


def user_features(orders_prior_u: pd.DataFrame, lines: pd.DataFrame, orders_train: pd.DataFrame) -> pd.DataFrame:
    """Thuộc tính cấp người dùng."""
    u = orders_prior_u.groupby("user_id").agg(
        u_n_orders=("order_number", "max"),
        u_mean_gap=("days_since_prior_order", "mean"),
        u_last_cum_days=("cum_days", "max"),
    )
    basket = lines.groupby(["user_id", "order_id"]).size().groupby("user_id").mean()
    u["u_mean_basket"] = basket
    # Đơn đầu tiên luôn có reordered = 0 nên loại ra để tỷ lệ không bị kéo xuống
    later = lines[lines["order_number"] > 1]
    u["u_reorder_ratio"] = later.groupby("user_id")["reordered"].mean()
    tgt = orders_train.set_index("user_id")["days_since_prior_order"]
    u["u_days_to_target"] = tgt.reindex(u.index)
    u["u_target_cum_days"] = u["u_last_cum_days"] + u["u_days_to_target"].fillna(0)
    return u.reset_index()


def user_product_features(lines: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    """Thuộc tính cấp cặp (người dùng, sản phẩm)."""
    up = lines.groupby(["user_id", "product_id"]).agg(
        up_n_bought=("order_id", "size"),
        up_last_order=("order_number", "max"),
        up_last_cum_days=("cum_days", "max"),
        up_mean_cart_pos=("add_to_cart_order", "mean"),
    ).reset_index()
    up = up.merge(users_df[["user_id", "u_n_orders", "u_target_cum_days"]], on="user_id", how="left")
    up["up_rate"] = up["up_n_bought"] / up["u_n_orders"]
    up["up_orders_since_last"] = up["u_n_orders"] - up["up_last_order"]
    up["up_days_since_last"] = up["u_target_cum_days"] - up["up_last_cum_days"]
    return up.drop(columns=["up_last_order", "up_last_cum_days", "u_n_orders", "u_target_cum_days"])


def catalog_reorder_rates(lines_fit: pd.DataFrame, products: pd.DataFrame, smoothing_m: float = 20.0) -> pd.DataFrame:
    """Tỷ lệ mua lại theo sản phẩm, gian hàng, ngành hàng, chỉ từ người dùng fit.

    Tỷ lệ của sản phẩm được làm trơn về tỷ lệ của gian hàng chứa nó:
        p_rate = (số dòng mua lại + m * tỷ lệ gian hàng) / (số dòng + m)
    để sản phẩm hiếm (ít dòng) không nhận tỷ lệ cực đoan 0 hoặc 1. Sản phẩm không xuất hiện ở người
    dùng fit nhận đúng tỷ lệ của gian hàng.
    """
    cat = products[["product_id", "aisle_id", "department_id"]]
    x = lines_fit[["product_id", "reordered"]].merge(cat, on="product_id", how="left")
    a_rate = x.groupby("aisle_id")["reordered"].mean().rename("a_reorder_rate")
    d_rate = x.groupby("department_id")["reordered"].mean().rename("d_reorder_rate")
    global_rate = x["reordered"].mean()
    p = x.groupby("product_id")["reordered"].agg(["sum", "size"])

    out = cat.copy()
    out = out.merge(a_rate, left_on="aisle_id", right_index=True, how="left")
    out = out.merge(d_rate, left_on="department_id", right_index=True, how="left")
    out["d_reorder_rate"] = out["d_reorder_rate"].fillna(global_rate)
    out["a_reorder_rate"] = out["a_reorder_rate"].fillna(out["d_reorder_rate"])
    out = out.merge(p, left_on="product_id", right_index=True, how="left")
    out[["sum", "size"]] = out[["sum", "size"]].fillna(0)
    out["p_reorder_rate"] = (out["sum"] + smoothing_m * out["a_reorder_rate"]) / (out["size"] + smoothing_m)
    return out[["product_id", "aisle_id", "department_id", "p_reorder_rate", "a_reorder_rate", "d_reorder_rate"]]


def build_feature_table(orders_prior: pd.DataFrame, op_prior: pd.DataFrame, orders_train: pd.DataFrame,
                        products: pd.DataFrame, pairs: pd.DataFrame, user_split: pd.DataFrame,
                        smoothing_m: float = 20.0) -> pd.DataFrame:
    """Ghép toàn bộ thuộc tính vào bảng cặp. Trả về bảng có cột split, label và các thuộc tính."""
    users = pairs["user_id"].unique()
    lines, orders_u = _prior_lines(orders_prior, op_prior, users)

    u = user_features(orders_u, lines, orders_train)
    up = user_product_features(lines, u)

    fit_users = user_split.loc[user_split["split"] == "fit", "user_id"].to_numpy()
    rates = catalog_reorder_rates(lines[lines["user_id"].isin(fit_users)], products, smoothing_m)

    df = pairs.merge(user_split, on="user_id", how="left")
    df = df.merge(up, on=["user_id", "product_id"], how="left")
    df = df.merge(u.drop(columns=["u_last_cum_days", "u_target_cum_days"]), on="user_id", how="left")
    df = df.merge(rates, on="product_id", how="left")

    # Kiểm tra: mọi cặp phải có thuộc tính cặp (vì cặp được tạo từ chính lịch sử prior)
    if df["up_n_bought"].isna().any():
        raise AssertionError("Có cặp không tìm thấy lịch sử prior, kiểm tra lại bước ghép.")
    forbidden = {"add_to_cart_order", "reordered", "eval_set"}
    leaked = forbidden & set(df.columns)
    if leaked:
        raise AssertionError(f"Cột có nguy cơ rò rỉ nằm trong bảng thuộc tính: {leaked}")

    float_cols = [c for c in FEATURE_DESCRIPTIONS if c in df.columns]
    df[float_cols] = df[float_cols].astype("float32")
    return df


def feature_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Thống kê mô tả thuộc tính theo nhãn, chỉ trên phần fit (dùng cho mục 4.6)."""
    fit = df[df["split"] == "fit"]
    desc = fit.groupby("label")[features].median().T
    desc.columns = ["trung_vi_nhan_0", "trung_vi_nhan_1"]
    desc["ty_le_thieu"] = fit[features].isna().mean()
    desc["mo_ta"] = [FEATURE_DESCRIPTIONS.get(f, "") for f in desc.index]
    return desc
