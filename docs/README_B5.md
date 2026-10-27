# Bước 5. Phân lớp dựa trên luật kết hợp

Người phụ trách: Cao Tấn Hùng (B5). Người duyệt: Quỳnh Trang (B4).

## 1. Định nghĩa bài toán và nhãn (task T34)

| Nội dung | Quy ước |
|---|---|
| Đơn vị quan sát | Một cặp (`user_id`, `product_id`) mà người dùng đã mua ít nhất một lần trong các đơn prior |
| Phạm vi người dùng | Chỉ người dùng có đơn thuộc `eval_set = train` (trong mẫu 10% của bước 2) |
| Nhãn | `label = 1` nếu sản phẩm xuất hiện trong đơn train của người dùng đó, ngược lại `label = 0` |
| Loại trừ | Người dùng thuộc `eval_set = test` (Instacart không công bố nhãn) |
| Kiểm tra chéo | Số cặp nhãn 1 phải bằng số dòng của đơn train có `reordered = 1` |

Lưu ý: nhãn này **khác** cột `reordered` của bảng prior. `reordered` cho biết một dòng trong quá khứ có phải là
mua lại hay không; nhãn bước 5 cho biết sản phẩm đã từng mua có được mua tiếp trong đơn kế tiếp hay không.
Mục 6 của EDA và bước rời rạc hóa của B2 cần dùng nhãn này.

Sản phẩm mới (chưa từng mua) xuất hiện trong đơn train không thuộc tập ứng viên, nên nằm ngoài bài toán. Tỷ lệ
của chúng được in ở mục A2 của notebook và cần nêu trong phần hạn chế.

## 2. Chia tập (task T38)

Chia theo `user_id`, hạt giống 42: 70% `fit` (huấn luyện), 10% `valid` (kiểm định), 20% `holdout` (kiểm tra).
Tên `fit / valid / holdout` được chọn để không nhầm với `eval_set` train/test của Instacart.

Chỉ học trên `fit`: tỷ lệ mua lại theo sản phẩm, gian hàng, ngành hàng; điểm cắt rời rạc hóa; luật lớp CBA và CMAR.
Chỉ chọn trên `valid`: độ sâu cây, số cây LightGBM, ngưỡng quyết định. `holdout` chỉ dùng một lần ở mục B8.

## 3. Tệp bàn giao (thư mục `data/processed/b5/`)

| Tệp | Người dùng | Nội dung |
|---|---|---|
| `b5_pairs_labels.parquet` | B1, B2 | `user_id`, `product_id`, `label`, `split` |
| `b5_user_split.parquet` | B2 | `user_id`, `split` |
| `b5_label_stats.json`, `b5_split_table.csv` | B1, TN | Số cặp, tỷ lệ nhãn 1, kiểm tra chéo |
| `b5_cutpoints.json`, `b5_discretization.csv` | B2 (mục 4.3) | Điểm cắt và tỷ lệ nhãn 1 trong từng khoảng |
| `b5_results_holdout.csv` | TN (Chương 5) | Bảng so sánh mô hình trên holdout, kèm khoảng tin cậy 95% |
| `b5_cba_rules.csv`, `b5_cmar_rules.csv` | B4, TN | Luật của bộ phân lớp |
| `b5_ablation_r4.csv`, `b5_r4_match.csv` | TN | Luật bước 4 có giúp dự đoán không |
| `b5_imbalance_valid.csv` | B5 | So sánh bản gốc, minsup theo lớp, bản cân bằng trên valid |
| `b5_cba_goc_rules.csv`, `b5_importance_vs_cba.csv` | B5 | Luật của CBA gốc; đối chiếu mức quan trọng LightGBM với luật CBA |
| `b5_error_*.csv`, `b5_cba_sensitivity.csv`, `b5_run_info.json` | B5 | Phân tích lỗi, độ nhạy, thông tin tái lập |
| `reports/figures/b5_*.png` | Báo cáo | Đường precision theo recall, ma trận nhầm lẫn, mức quan trọng thuộc tính |

## 4. Cách chạy

1. Chép `src/b5_classification/` và `notebooks/05_classification.ipynb` vào repo.
2. Mở notebook trên Colab, sửa `REPO_DIR` trong ô Cấu hình cho đúng thư mục Drive của nhóm.
3. Nếu B4 đã xuất luật cấp gian hàng, đặt tệp vào `data/processed/rules_aisle.csv` (định dạng `association_rules`
   của mlxtend, hoặc cột `antecedents`, `consequents`, `lift` với tên gian hàng hoặc mã gian hàng).
   Tệp `T32_filtered_rules_aisle.csv` của Quỳnh Trang đọc trực tiếp được (dạng `aisle_24`). Chưa có thì notebook dùng luật dự phòng và ghi rõ trong `b5_run_info.json`.
4. Chạy Phần A trước (vài phút), gửi tệp bàn giao cho B1, B2. Sau đó chạy Phần B.
5. Kiểm thử đơn vị: `python -m pytest tests -q`.

Thư viện: pandas, numpy, scipy, scikit-learn, lightgbm, pyarrow, matplotlib. Không cần pyarc hay mlxtend.

## 5. Mất cân bằng nhãn (kết quả lần chạy 17/10/2026)

Nhãn 1 chỉ chiếm 9,74% số cặp. Với cấu hình gốc của bài báo (minsup 1%, minconf 50%), bộ phân lớp CBA giữ 0 luật lớp 1 và CMAR luôn đoán 0. Notebook vì vậy chạy song song:

- **Bản gốc**: đúng cấu hình bài báo, giữ để đối chứng.
- **Minsup theo lớp** (Liu, Ma & Wong, 2000): chỉ so trên valid; không cải thiện (vẫn 0 luật lớp 1).
- **Bản cân bằng** (mô hình chính): lấy mẫu dưới lớp 0 trên phần fit (giữ mọi cặp nhãn 1, lấy cùng số cặp nhãn 0, hạt giống 42). Valid và holdout giữ phân phối thật.

Confidence của luật trong bản cân bằng phải quy đổi về phân phối thật: precision ≈ c / [c + (1 − c) · k], với k là tỷ số nhãn 0 trên nhãn 1 của phần fit (k = 9,22).

## 6. Quyết định về CMAR (task T36)

Đề xuất: **làm CMAR**, vì CMAR dùng chung bộ đếm luật với CBA nên chỉ thêm bước cắt tỉa và bước phân lớp.
Cả nhóm chốt trước ngày đóng băng code.

## 7. Những điểm khác với bài báo gốc (phải nêu trong báo cáo)

| Điểm | Bài báo | Cài đặt của nhóm | Lý do |
|---|---|---|---|
| Đếm luật CBA | Apriori sinh ứng viên theo mức | Đếm trực tiếp mọi condset của từng tổ hợp thuộc tính | Cho đúng cùng tập ruleitem phổ biến, nhanh hơn khi dữ liệu nằm trong bộ nhớ |
| Dựng bộ phân lớp CBA | Đề xuất M2 cho dữ liệu trên đĩa | M1 viết dạng vectơ | Dữ liệu vừa bộ nhớ; M1 minh bạch, dễ kiểm tra |
| Khai thác luật CMAR | Biến thể FP-growth, cây CR-tree | Dùng chung bộ đếm với CBA | Hai cấu trúc chỉ phục vụ tốc độ và bộ nhớ, không đổi tập luật |
| Rời rạc hóa | Phương pháp entropy của Fayyad và Irani | Cùng tiêu chuẩn MDL, thêm giới hạn tối đa 5 khoảng | Hàng trăm nghìn dòng làm ngưỡng MDL rất nhỏ, sinh quá nhiều khoảng |
| Mức ý nghĩa chi bình phương của CMAR | Không nêu | 5% (ngưỡng 3,841) | Mức thông dụng |
| Ngưỡng chênh lệch confidence 20% của CMAR | Nêu giá trị, không mô tả cơ chế | Không dùng | Định nghĩa nằm trong luận văn của W. Li (2001), không có trong bài báo |
| Không luật nào khớp (CMAR) | Không nêu | Gán lớp đa số của dữ liệu huấn luyện | Cần một quy tắc mặc định |
| Dữ liệu học luật | Toàn bộ dữ liệu huấn luyện | Thêm bản học trên phần fit cân bằng (lấy mẫu dưới lớp 0) | Bản gốc không giữ được luật lớp 1 nào khi nhãn 1 chỉ 9,74% |
| Điểm số để vẽ đường precision theo recall | Không có (hai thuật toán chỉ trả nhãn) | CBA: tỷ lệ lớp 1 trong các dòng khớp luật quyết định; CMAR: tỷ trọng chi bình phương có trọng số của nhóm lớp 1 | Cần điểm số để tính PR-AUC |
