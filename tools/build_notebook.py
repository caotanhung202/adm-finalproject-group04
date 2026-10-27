"""Sinh notebooks/05_classification.ipynb từ danh sách ô dưới đây (giữ notebook gọn, dễ xem khác biệt)."""

from pathlib import Path

import nbformat as nbf

C = []  # (loại ô, nội dung)


def md(s):
    C.append(("md", s.strip("\n")))


def code(s):
    C.append(("code", s.strip("\n")))


md(r"""
# Bước 5. Phân lớp dựa trên luật kết hợp (CBA, CMAR)

**Câu hỏi của bước 5.** Với mỗi cặp (người dùng, sản phẩm) mà người dùng đã từng mua trong các đơn prior,
dự đoán sản phẩm đó có xuất hiện trong đơn kế tiếp (đơn train) hay không, bằng phân lớp dựa trên luật kết hợp
(CBA làm mô hình chính, CMAR mở rộng), so với cây quyết định tiêu chí entropy và LightGBM.

**Cách chạy.** Notebook có hai phần.
- **Phần A (chạy trước, nhanh):** tạo bảng nhãn và chia người dùng, xuất Parquet để bàn giao cho B1 (đo mất cân bằng nhãn) và B2 (rời rạc hóa theo nhãn mới).
- **Phần B:** thuộc tính, rời rạc hóa, CBA, CMAR, mô hình đối chứng, đánh giá trên holdout, phân tích lỗi.

**Nguyên tắc chống rò rỉ dữ liệu** (task T45 sẽ kiểm tra độc lập):
1. Nhãn chỉ lấy từ đơn train; thuộc tính chỉ lấy từ đơn prior (cộng khoảng cách ngày tới đơn cần dự đoán).
2. Chia theo `user_id`: một người dùng chỉ thuộc một trong ba phần fit, valid, holdout.
3. Tỷ lệ mua lại theo sản phẩm, gian hàng, ngành hàng; điểm cắt rời rạc hóa; luật lớp: chỉ học trên phần fit.
4. Siêu tham số và ngưỡng quyết định: chọn trên valid. Holdout chỉ dùng **một lần** ở mục B8.
""")

md("## 0. Cấu hình")
code(r"""
import os, sys, json, time, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

# SỬA ĐƯỜNG DẪN NÀY cho đúng thư mục chung của nhóm trên Drive
REPO_DIR = Path(os.environ.get("B5_REPO_DIR", "/content/drive/MyDrive/Nhom04_ADM/repo"))
DATA_DIR = Path(os.environ.get("B5_DATA_DIR", REPO_DIR / "data" / "processed"))     # 5 tệp Parquet của B2
OUT_DIR = Path(os.environ.get("B5_OUT_DIR", REPO_DIR / "data" / "processed" / "b5"))
FIG_DIR = Path(os.environ.get("B5_FIG_DIR", REPO_DIR / "reports" / "figures"))
# Tệp luật cấp gian hàng do B4 xuất (để None nếu chưa có: notebook sẽ tự khai thác luật dự phòng)
RULES4_PATH = os.environ.get("B5_RULES4_PATH", str(REPO_DIR / "data" / "processed" / "rules_aisle.csv"))

sys.path.insert(0, str(Path(os.environ.get("B5_SRC_DIR", REPO_DIR / "src"))))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

PARAMS = {
    "holdout_frac": 0.20, "valid_frac": 0.10,          # 70% fit, 10% valid, 20% holdout theo người dùng
    "max_intervals": 5,                                # số khoảng tối đa mỗi thuộc tính (xem discretize.py)
    "cba_min_support": 0.01, "cba_min_confidence": 0.50, "cba_max_len": 3, "cba_prune": True,
    "cmar_coverage_delta": 4,
    "run_cmar": True,
    "n_boot": 200,
}
print("Python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__)
print("DATA_DIR =", DATA_DIR)
""")

md(r"""
# PHẦN A. Nhãn và chia tập (bàn giao cho B1, B2)

## A1. Đọc dữ liệu của bước 2 và kiểm tra đầu vào
Mọi dòng có cột "Kỳ vọng" phải **Đạt**. Đặc biệt chú ý dòng *Đơn prior không có dòng sản phẩm nào*: bảng
`order_products_prior_sample` có đúng 2.066.000 dòng (một số tròn), nên cần xác nhận bảng không bị cắt bớt.
Nếu dòng này khác 0, báo ngay cho Thiên Trang trước khi đi tiếp.
""")
code(r"""
from b5_classification import labels as L

d = L.load_b2_data(DATA_DIR)
for name in ["orders_prior", "orders_train", "op_prior", "op_train", "products"]:
    print(f"{name:13s}", getattr(d, name).shape)
integrity = L.integrity_report(d)
integrity
""")

md(r"""
## A2. Tạo bảng cặp và nhãn (task T34)
- Đối tượng: mỗi cặp (`user_id`, `product_id`) người dùng đã mua trong đơn prior, chỉ lấy người dùng có đơn train.
- Nhãn bằng 1 nếu sản phẩm có trong đơn train của người dùng đó.
- **Kiểm tra chéo:** số cặp nhãn 1 phải bằng số dòng của đơn train có `reordered = 1`.
- `ty_le_dong_train_la_san_pham_moi`: phần sản phẩm trong đơn train mà người dùng chưa từng mua. Các sản phẩm này
  nằm ngoài tập ứng viên, nên mô hình không thể "đoán" ra chúng; cần nêu là giới hạn của bài toán.
""")
code(r"""
pairs, label_stats = L.build_labels(d)
assert label_stats["kiem_tra_nhan_1_bang_reordered_train"], "Số nhãn 1 không khớp reordered của đơn train"
pd.Series(label_stats, name="giá trị").to_frame()
""")

md("## A3. Chia người dùng thành fit, valid, holdout (task T38)")
code(r"""
user_split = L.split_users(pairs["user_id"].to_numpy(), PARAMS["holdout_frac"], PARAMS["valid_frac"], SEED)
L.assert_no_user_overlap(user_split)
split_table = L.split_summary(pairs, user_split)
split_table
""")

md(r"""
## A4. Xuất tệp bàn giao
- `b5_pairs_labels.parquet`: `user_id`, `product_id`, `label`, `split` → B1 dùng đo mất cân bằng (thay cho `reordered` ở mục 6 của EDA).
- `b5_user_split.parquet`: `user_id`, `split` → B2 chỉ học điểm cắt trên người dùng có `split == "fit"`.
- `b5_label_stats.json`: các con số ở A2, A3 để viết báo cáo.
""")
code(r"""
pairs_out = pairs.merge(user_split, on="user_id", how="left")
pairs_out.to_parquet(OUT_DIR / "b5_pairs_labels.parquet", index=False)
user_split.to_parquet(OUT_DIR / "b5_user_split.parquet", index=False)
with open(OUT_DIR / "b5_label_stats.json", "w", encoding="utf-8") as fh:
    json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else (bool(v) if isinstance(v, (bool, np.bool_)) else int(v)))
               for k, v in label_stats.items()}, fh, ensure_ascii=False, indent=2)
split_table.to_csv(OUT_DIR / "b5_split_table.csv")
print("Đã xuất vào", OUT_DIR)
""")

md(r"""
# PHẦN B. Mô hình hóa

## B1. Thuộc tính cho từng cặp (task T37)
Thuộc tính chỉ lấy từ đơn prior. Tỷ lệ mua lại theo sản phẩm, gian hàng, ngành hàng chỉ tính trên người dùng fit.
""")
code(r"""
from b5_classification import features as F

df = F.build_feature_table(d.orders_prior, d.op_prior, d.orders_train, d.products, pairs, user_split)
BASE_FEATURES = list(F.FEATURE_DESCRIPTIONS)
print(df.shape)
F.feature_summary(df, BASE_FEATURES)
""")

md(r"""
## B2. Nối luật của bước 4 thành thuộc tính `r4_lift`
Nếu đã có tệp luật cấp gian hàng của B4 (`RULES4_PATH`), dùng luật đó. Nếu chưa có, khai thác luật **dự phòng**
trên đơn prior của người dùng fit (minsup 2%, minconf 50%, lift > 1) và ghi rõ trong báo cáo là luật dự phòng.
""")
code(r"""
from b5_classification import step4_link as S4

if RULES4_PATH and Path(RULES4_PATH).exists():
    rules4 = S4.load_step4_rules(RULES4_PATH, d.products, level="aisle")
    rules4_source = "B4"
else:
    fit_users = user_split.loc[user_split.split == "fit", "user_id"]
    fit_orders = d.orders_prior[d.orders_prior.user_id.isin(fit_users)]["order_id"]
    baskets = (d.op_prior[d.op_prior.order_id.isin(fit_orders)]
               .merge(d.products[["product_id", "aisle_id"]], on="product_id")[["order_id", "aisle_id"]]
               .drop_duplicates())
    rules4 = S4.mine_aisle_rules(baskets, "aisle_id", min_support=0.02, min_confidence=0.5)
    rules4_source = "dự phòng (tự khai thác trên người dùng fit)"
print("Nguồn luật:", rules4_source, "| số luật dùng được:", len(rules4))
df["r4_lift"] = S4.r4_lift_feature(df[["user_id", "product_id"]], d.orders_prior, d.op_prior, d.products, rules4).to_numpy()
FEATURES = BASE_FEATURES + ["r4_lift"]
print("Tỷ lệ cặp có ít nhất một luật khớp:", (df["r4_lift"] > 1).mean().round(4))
df.groupby(df["r4_lift"] > 1)["label"].agg(["size", "mean"]).rename(index={False: "không khớp luật", True: "khớp luật"})
""")

md(r"""
## B3. Rời rạc hóa theo entropy (Fayyad và Irani), chỉ học trên phần fit
Mỗi thuộc tính có tối đa `max_intervals` khoảng. Điểm cắt được lưu ra `b5_cutpoints.json` để B2 trình bày ở mục 4.3.
""")
code(r"""
from b5_classification.discretize import Discretizer

fit = df[df.split == "fit"].reset_index(drop=True)
valid = df[df.split == "valid"].reset_index(drop=True)
hold = df[df.split == "holdout"].reset_index(drop=True)
y_fit, y_valid, y_hold = fit.label.to_numpy(), valid.label.to_numpy(), hold.label.to_numpy()

disc = Discretizer(max_intervals=PARAMS["max_intervals"]).fit(fit, FEATURES, y_fit)
disc.to_json(OUT_DIR / "b5_cutpoints.json")
Xc_fit, Xc_valid, Xc_hold = disc.transform(fit), disc.transform(valid), disc.transform(hold)
LABELERS = {f: disc.interval_labels(f) for f in FEATURES}
disc_table = disc.summary(fit, y_fit)
disc_table.to_csv(OUT_DIR / "b5_discretization.csv", index=False)
print({f: len(c) + 1 for f, c in disc.cuts.items()})
disc_table
""")

md(r"""
## B4. CBA theo cấu hình gốc của bài báo (task T39)
Cấu hình đúng như Liu, Hsu và Ma (1998): minsup 1%, minconf 50%, cắt tỉa lỗi bi quan, học trên toàn bộ phần fit.
Ô in ra số luật lớp 1 trong bộ phân lớp: với nhãn 1 hiếm, con số này cho biết CBA gốc có học được luật cho lớp cần dự đoán hay không.
""")
code(r"""
from b5_classification.cba import CBAClassifier
from b5_classification.evaluation import classification_metrics

def n_class1(rules, classes):
    return int(sum(classes[r["cls_idx"]] == 1 for r in rules))

def cba_summary(m):
    return {"so_condset_pho_bien": m.n_condsets_, "so_CAR": m.n_cars_, "so_luat_sau_cat_tia": m.n_rules_after_prune_,
            "so_luat_trong_bo_phan_lop": len(m.rules_), "so_luat_lop_1_trong_bo_phan_lop": n_class1(m.rules_, m.classes_),
            "lop_mac_dinh": int(m.classes_[m.default_idx_]), **{k: round(v, 2) for k, v in m.timing_.items()}}

cba = CBAClassifier(PARAMS["cba_min_support"], PARAMS["cba_min_confidence"], PARAMS["cba_max_len"],
                    prune=PARAMS["cba_prune"]).fit(Xc_fit, y_fit)
cba_info = cba_summary(cba)
print(cba_info)
print("Valid:", {k: round(v, 4) for k, v in classification_metrics(y_valid, cba.predict(Xc_valid), cba.predict_score(Xc_valid)).items()})
""")

md(r"""
## B4b. Xử lý mất cân bằng nhãn
CBA chọn luật để giảm tổng số lỗi trên dữ liệu huấn luyện. Khi khoảng 90% số cặp có nhãn 0, phương án ít lỗi nhất là
dùng luật lớp 0 và để lớp 1 làm lớp mặc định, nên bộ phân lớp gốc gần như không có luật cho lớp 1. Hai cách xử lý được thử:

1. **minsup theo lớp** (Liu, Ma & Wong, 2000): ngưỡng hỗ trợ của lớp c bằng minsup × tỷ lệ của lớp c.
2. **Cân bằng lại phần fit bằng lấy mẫu dưới lớp 0** (giữ mọi cặp nhãn 1, lấy ngẫu nhiên cùng số cặp nhãn 0, hạt giống 42).
   Chỉ phần fit bị lấy mẫu; valid và holdout giữ nguyên phân phối thật. Điểm cắt rời rạc hóa vẫn học trên toàn bộ phần fit.

Cả hai được so trên **valid**; phương án tốt hơn trở thành mô hình CBA chính ở các bước sau.
""")
code(r"""
rng_bal = np.random.default_rng(SEED)
pos_idx, neg_idx = np.where(y_fit == 1)[0], np.where(y_fit == 0)[0]
bal_idx = np.sort(np.concatenate([pos_idx, rng_bal.choice(neg_idx, size=len(pos_idx), replace=False)]))
Xc_bal, y_bal = Xc_fit.iloc[bal_idx].reset_index(drop=True), y_fit[bal_idx]
print("Phần fit cân bằng:", len(y_bal), "cặp, tỷ lệ nhãn 1 =", round(y_bal.mean(), 4))

cba_ms = CBAClassifier(PARAMS["cba_min_support"], PARAMS["cba_min_confidence"], PARAMS["cba_max_len"],
                       prune=PARAMS["cba_prune"], class_minsup=True).fit(Xc_fit, y_fit)
cba_bal = CBAClassifier(PARAMS["cba_min_support"], PARAMS["cba_min_confidence"], PARAMS["cba_max_len"],
                        prune=PARAMS["cba_prune"]).fit(Xc_bal, y_bal)
cba_ms_info, cba_bal_info = cba_summary(cba_ms), cba_summary(cba_bal)
from b5_classification.baselines import best_f1_threshold

imb = []
for name, m in [("CBA gốc", cba), ("CBA minsup theo lớp", cba_ms), ("CBA cân bằng", cba_bal)]:
    s = m.predict_score(Xc_valid)
    thr, f1_thr = best_f1_threshold(y_valid, s)
    met = classification_metrics(y_valid, m.predict(Xc_valid), s)
    imb.append({"cau_hinh": name, "so_luat_bo_phan_lop": len(m.rules_), "so_luat_lop_1": n_class1(m.rules_, m.classes_),
                "f1_nhan_goc": met["f1"], "f1_nguong_valid": f1_thr, "pr_auc": met["pr_auc"], "roc_auc": met["roc_auc"]})
imb = pd.DataFrame(imb)
imb.to_csv(OUT_DIR / "b5_imbalance_valid.csv", index=False)
imb
""")

md("### Độ nhạy theo minsup và cắt tỉa (đánh giá trên valid, không đụng holdout)")
code(r"""
sens = []
for data_name, Xtr, ytr in [("gốc", Xc_fit, y_fit), ("cân bằng", Xc_bal, y_bal)]:
    for ms in [0.005, 0.01, 0.02]:
        for pr in [False, True]:
            m = CBAClassifier(ms, 0.5, PARAMS["cba_max_len"], prune=pr).fit(Xtr, ytr)
            s = m.predict_score(Xc_valid)
            met = classification_metrics(y_valid, m.predict(Xc_valid), s)
            sens.append({"du_lieu_huan_luyen": data_name, "minsup": ms, "cat_tia": pr, "so_CAR": m.n_cars_,
                         "so_luat_sau_cat_tia": m.n_rules_after_prune_, "so_luat_bo_phan_lop": len(m.rules_),
                         "f1_nguong_valid": best_f1_threshold(y_valid, s)[1], "pr_auc": met["pr_auc"],
                         "thoi_gian_s": sum(m.timing_.values())})
sens = pd.DataFrame(sens)
sens.to_csv(OUT_DIR / "b5_cba_sensitivity.csv", index=False)
sens
""")

md("## B5. CMAR (mở rộng, task T41): cấu hình gốc và bản học trên phần fit cân bằng")
code(r"""
from b5_classification.cmar import CMARClassifier

def cmar_summary(m):
    return {"so_luat_ung_vien": m.n_candidate_rules_, "sau_cat_tia_luat_tong_quat": m.n_after_general_,
            "sau_kiem_dinh_chi_binh_phuong": m.n_after_chi2_, "sau_do_phu_csdl": m.n_rules_,
            "so_luat_lop_1": int((m.rules_["cls_idx"] == int(np.where(m.classes_ == 1)[0][0])).sum()),
            "thoi_gian_s": round(m.timing_["total_fit_s"], 2)}

cmar = cmar_bal = None
cmar_info = cmar_bal_info = None
if PARAMS["run_cmar"]:
    args = (PARAMS["cba_min_support"], PARAMS["cba_min_confidence"], PARAMS["cba_max_len"])
    cmar = CMARClassifier(*args, coverage_delta=PARAMS["cmar_coverage_delta"]).fit(Xc_fit, y_fit)
    cmar_bal = CMARClassifier(*args, coverage_delta=PARAMS["cmar_coverage_delta"]).fit(Xc_bal, y_bal)
    cmar_info, cmar_bal_info = cmar_summary(cmar), cmar_summary(cmar_bal)
    print("CMAR gốc:", cmar_info)
    print("CMAR cân bằng:", cmar_bal_info)
""")

md("## B6. Mô hình đối chứng (task T40)")
code(r"""
from b5_classification import baselines as B

tree_disc, tree_disc_grid, t_tree_disc = B.fit_decision_tree(Xc_fit, y_fit, Xc_valid, y_valid, seed=SEED)
Xr_fit, Xr_valid, Xr_hold = (x[FEATURES].fillna(-1) for x in (fit, valid, hold))
tree_raw, tree_raw_grid, t_tree_raw = B.fit_decision_tree(Xr_fit, y_fit, Xr_valid, y_valid, seed=SEED)
lgbm, t_lgbm = B.fit_lightgbm(fit[FEATURES], y_fit, valid[FEATURES], y_valid, seed=SEED)
one_rule = B.SingleFeatureRule("up_rate").fit(valid, y_valid)
print("Cây (mã khoảng) chọn max_depth =", tree_disc.get_depth(), "| Cây (gốc) max_depth =", tree_raw.get_depth(),
      "| LightGBM số cây =", lgbm.best_iteration_, "| ngưỡng up_rate =", round(one_rule.threshold_, 4))
pd.concat([tree_disc_grid.assign(du_lieu="mã khoảng"), tree_raw_grid.assign(du_lieu="giá trị gốc")])
""")

md(r"""
## B7. Chọn ngưỡng quyết định trên valid
CBA và CMAR có nhãn dự đoán riêng (luật đầu tiên khớp, hoặc nhóm luật mạnh nhất). Báo cáo cả **nhãn gốc** của thuật
toán và **ngưỡng theo F1** chọn trên valid áp lên điểm số. Một điểm số gần như hằng số trên valid (ít hơn 3 giá trị khác
nhau) không có ngưỡng ý nghĩa, nên chỉ báo cáo nhãn gốc.
""")
code(r"""
def score_fns():
    fns = {
        "CBA gốc": (lambda X, Xc: cba.predict_score(Xc)),
        "CBA cân bằng": (lambda X, Xc: cba_bal.predict_score(Xc)),
        "Cây entropy (mã khoảng)": (lambda X, Xc: tree_disc.predict_proba(Xc)[:, 1]),
        "Cây entropy (giá trị gốc)": (lambda X, Xc: tree_raw.predict_proba(X[FEATURES].fillna(-1))[:, 1]),
        "LightGBM": (lambda X, Xc: lgbm.predict_proba(X[FEATURES])[:, 1]),
        "Luật một thuộc tính (up_rate)": (lambda X, Xc: one_rule.predict_score(X)),
    }
    if cmar is not None:
        fns["CMAR gốc"] = lambda X, Xc: cmar.predict_score(Xc)
        fns["CMAR cân bằng"] = lambda X, Xc: cmar_bal.predict_score(Xc)
    return fns

LABEL_MODELS = {"CBA gốc": cba, "CBA cân bằng": cba_bal, "CMAR gốc": cmar, "CMAR cân bằng": cmar_bal}
thresholds, degenerate = {}, set()
for name, fn in score_fns().items():
    s_val = fn(valid, Xc_valid)
    if len(np.unique(np.round(s_val, 6))) < 3:
        degenerate.add(name)
        continue
    thresholds[name] = B.best_f1_threshold(y_valid, s_val)[0]
print("Điểm số suy biến (chỉ báo cáo nhãn gốc):", degenerate or "không có")
pd.Series(thresholds, name="ngưỡng chọn trên valid")
""")

md(r"""
## B8. Đánh giá cuối trên holdout (chạy MỘT lần, sau khi mọi lựa chọn đã cố định)
Khoảng tin cậy 95% bằng bootstrap theo người dùng (`n_boot` lần).
""")
code(r"""
from b5_classification.evaluation import bootstrap_ci, format_vn

users_hold = hold.user_id.to_numpy()
results, preds, scores = [], {}, {}
for name, fn in score_fns().items():
    s = fn(hold, Xc_hold)
    variants = {}
    if name in LABEL_MODELS:
        variants[name + " (nhãn gốc)"] = LABEL_MODELS[name].predict(Xc_hold).astype(int)
    if name in thresholds:
        variants[name + " (ngưỡng valid)"] = (s >= thresholds[name]).astype(int)
    scores[name] = s
    for vname, p in variants.items():
        met = classification_metrics(y_hold, p, s)
        ci = bootstrap_ci(y_hold, p, s, users_hold, PARAMS["n_boot"], SEED).set_index("do_do")
        met.update({"f1_ci95": f"[{ci.loc['f1','can_duoi']:.3f}; {ci.loc['f1','can_tren']:.3f}]".replace(".", ","),
                    "pr_auc_ci95": f"[{ci.loc['pr_auc','can_duoi']:.3f}; {ci.loc['pr_auc','can_tren']:.3f}]".replace(".", ",")})
        results.append({"mo_hinh": vname, **met})
        preds[vname] = p
majority = np.zeros_like(y_hold)
results.append({"mo_hinh": "Luôn đoán 0 (lớp đa số)", **classification_metrics(y_hold, majority)})
results = pd.DataFrame(results)
results.to_csv(OUT_DIR / "b5_results_holdout.csv", index=False)
format_vn(results.drop(columns=["ty_le_du_doan_1"]), 4)
""")

code(r"""
from b5_classification.evaluation import plot_pr_curves, plot_confusion

pr_models = [k for k in ["CBA gốc", "CBA cân bằng", "CMAR cân bằng", "Cây entropy (mã khoảng)", "LightGBM",
                         "Luật một thuộc tính (up_rate)"] if k in scores]
plot_pr_curves(y_hold, {k: scores[k] for k in pr_models}, FIG_DIR / "b5_pr_curves.png",
               "Đường precision theo recall trên tập holdout")
plt.show()
cm_models = {k: preds[k] for k in ["CBA gốc (nhãn gốc)", "CBA cân bằng (ngưỡng valid)", "LightGBM (ngưỡng valid)"] if k in preds}
plot_confusion(y_hold, cm_models, FIG_DIR / "b5_confusion.png")
plt.show()
plt.close("all")
""")

md(r"""
## B9. Luật bước 4 có giúp dự đoán không? (ablation)
So sánh cùng một mô hình khi **có** và **không có** thuộc tính `r4_lift`, chênh lệch tính bằng bootstrap ghép cặp
theo người dùng trên holdout. Khoảng 95% không chứa 0 thì khác biệt có ý nghĩa. Dùng CBA cân bằng (mô hình chính) và LightGBM.
""")
code(r"""
from b5_classification.evaluation import paired_bootstrap_diff

no_r4 = [f for f in FEATURES if f != "r4_lift"]
cba_no = CBAClassifier(PARAMS["cba_min_support"], PARAMS["cba_min_confidence"], PARAMS["cba_max_len"],
                       prune=PARAMS["cba_prune"]).fit(Xc_bal[no_r4], y_bal)
t_no = B.best_f1_threshold(y_valid, cba_no.predict_score(Xc_valid[no_r4]))[0]
s_no = cba_no.predict_score(Xc_hold[no_r4])
lgb_no, _ = B.fit_lightgbm(fit[no_r4], y_fit, valid[no_r4], y_valid, seed=SEED)
tl_no = B.best_f1_threshold(y_valid, lgb_no.predict_proba(valid[no_r4])[:, 1])[0]
sl_no = lgb_no.predict_proba(hold[no_r4])[:, 1]

ablation = pd.concat([
    paired_bootstrap_diff(y_hold, users_hold,
                          {"pred": preds["CBA cân bằng (ngưỡng valid)"], "score": scores["CBA cân bằng"]},
                          {"pred": (s_no >= t_no).astype(int), "score": s_no}, PARAMS["n_boot"], SEED).assign(mo_hinh="CBA cân bằng"),
    paired_bootstrap_diff(y_hold, users_hold,
                          {"pred": preds["LightGBM (ngưỡng valid)"], "score": scores["LightGBM"]},
                          {"pred": (sl_no >= tl_no).astype(int), "score": sl_no}, PARAMS["n_boot"], SEED).assign(mo_hinh="LightGBM"),
])
ablation["so_luat_CBA_co_r4"] = sum(any(cba_bal.feature_names_[a] == "r4_lift" for a, _ in r["items"]) for r in cba_bal.rules_)
ablation.to_csv(OUT_DIR / "b5_ablation_r4.csv", index=False)
r4_match = pd.DataFrame({"khop": df["r4_lift"] > 1, "label": df["label"]}).groupby("khop")["label"].agg(["size", "mean"])
r4_match.to_csv(OUT_DIR / "b5_r4_match.csv")
display(r4_match)
ablation
""")

md(r"""
## B10. Đọc luật CBA (task T43)
Luật của mô hình chính (CBA cân bằng), mỗi luật kèm số dòng và độ chính xác **trên holdout** (luật có còn đúng trên người dùng chưa thấy hay không).
Lưu ý: confidence của CBA cân bằng tính trên phần fit đã cân bằng (tỷ lệ nhãn 1 là 50%), nên không so trực tiếp với tỷ lệ thật.
""")
code(r"""
def rules_with_holdout(m, path):
    fired = m.fired_rule(Xc_hold)
    rt = m.rules_table(LABELERS)
    hs = []
    for k in range(len(m.rules_) + 1):
        msk = fired == (k if k < len(m.rules_) else -1)
        cls = m.rules_[k]["cls_idx"] if k < len(m.rules_) else m.default_idx_
        hs.append({"so_dong_holdout_luat_quyet_dinh": int(msk.sum()),
                   "do_chinh_xac_tren_holdout": float((y_hold[msk] == m.classes_[cls]).mean()) if msk.any() else np.nan})
    rt = pd.concat([rt, pd.DataFrame(hs)], axis=1)
    rt.to_csv(path, index=False)
    return rt

rules_with_holdout(cba, OUT_DIR / "b5_cba_goc_rules.csv")
rt = rules_with_holdout(cba_bal, OUT_DIR / "b5_cba_rules.csv")
if cmar_bal is not None:
    cmar_bal.rules_table(LABELERS, cba_bal.describe_condset).to_csv(OUT_DIR / "b5_cmar_rules.csv", index=False)
rt[(rt["lop"] == 1) & (rt["so_dong_holdout_luat_quyet_dinh"] >= 100)].head(15)
""")

md("## B11. Phân tích lỗi theo nhóm người dùng, nhóm sản phẩm (CBA cân bằng so với LightGBM)")
code(r"""
from b5_classification.evaluation import error_by_group

dept_name = hold[["product_id"]].merge(d.products[["product_id", "department"]], on="product_id", how="left")["department"]
groups = {
    "nhom_so_don_prior": pd.cut(hold["u_n_orders"], [0, 5, 10, 20, 50, 100], labels=["3-5", "6-10", "11-20", "21-50", "51-99"]),
    "so_lan_da_mua": pd.cut(hold["up_n_bought"], [0, 1, 2, 5, 1000], labels=["1", "2", "3-5", "6+"]),
    "nganh_hang": dept_name,
}
err_tables = {}
for gname, g in groups.items():
    t = error_by_group(y_hold, preds["CBA cân bằng (ngưỡng valid)"], g, gname).add_prefix("CBA_").rename(columns={f"CBA_{gname}": gname})
    t2 = error_by_group(y_hold, preds["LightGBM (ngưỡng valid)"], g, gname)[["f1"]].rename(columns={"f1": "LightGBM_f1"})
    err_tables[gname] = pd.concat([t, t2], axis=1)
    err_tables[gname].to_csv(OUT_DIR / f"b5_error_{gname}.csv", index=False)
err_tables["nhom_so_don_prior"]
""")
code(r"""
err_tables["so_lan_da_mua"]
""")
code(r"""
err_tables["nganh_hang"].sort_values("CBA_so_cap", ascending=False).head(12)
""")

md("## B12. Mức quan trọng thuộc tính của LightGBM (để đối chiếu với các thuộc tính xuất hiện trong luật CBA)")
code(r"""
plt.close("all")                      # tránh vẽ chồng lên hình của ô trước
imp = pd.Series(lgbm.booster_.feature_importance("gain"), index=FEATURES).sort_values()
fig_imp, ax = plt.subplots(figsize=(6, 4.5))
(imp / imp.sum()).plot.barh(ax=ax, color="#3b6ea8")
ax.set_xlabel("Tỷ trọng mức giảm hàm mất mát (gain)")
fig_imp.tight_layout(); fig_imp.savefig(FIG_DIR / "b5_lgbm_importance.png", dpi=200); plt.show()
in_cba = pd.Series({f: sum(any(cba_bal.feature_names_[a] == f for a, _ in r["items"]) for r in cba_bal.rules_) for f in FEATURES},
                   name="so_luat_CBA_chua_thuoc_tinh")
imp_tab = pd.concat([(imp / imp.sum()).rename("ty_trong_gain_LightGBM"), in_cba], axis=1).sort_values("ty_trong_gain_LightGBM", ascending=False)
imp_tab.to_csv(OUT_DIR / "b5_importance_vs_cba.csv")
imp_tab
""")

md("## B13. Lưu thông tin lần chạy (để tái lập)")
code(r"""
import sklearn, scipy
try:
    import lightgbm; lgb_ver = lightgbm.__version__
except Exception:
    lgb_ver = None
run_info = {
    "thoi_diem": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "params": PARAMS,
    "nguon_luat_buoc_4": rules4_source, "so_luat_buoc_4_dung": int(len(rules4)),
    "phien_ban": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                  "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "lightgbm": lgb_ver},
    "so_cap_fit_can_bang": int(len(y_bal)),
    "cba_goc": cba_info, "cba_minsup_theo_lop": cba_ms_info, "cba_can_bang": cba_bal_info,
    "cmar_goc": cmar_info, "cmar_can_bang": cmar_bal_info,
    "lightgbm_so_cay": int(lgbm.best_iteration_), "cay_ma_khoang_do_sau": int(tree_disc.get_depth()),
    "cay_goc_do_sau": int(tree_raw.get_depth()),
    "thoi_gian_huan_luyen_s": {"cay_ma_khoang": t_tree_disc, "cay_goc": t_tree_raw, "lightgbm": t_lgbm},
    "nguong_valid": thresholds, "diem_so_suy_bien": sorted(degenerate),
}
with open(OUT_DIR / "b5_run_info.json", "w", encoding="utf-8") as fh:
    json.dump(run_info, fh, ensure_ascii=False, indent=2, default=float)
print(json.dumps(run_info, ensure_ascii=False, indent=1, default=float)[:2500])
""")

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in C]
out = Path(__file__).with_name("05_classification.ipynb")
nbf.write(nb, out)
print("Đã ghi", out)
