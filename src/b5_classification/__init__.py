"""Bước 5 của đồ án: phân lớp dựa trên luật kết hợp (CBA, CMAR) trên dữ liệu Instacart.

Các module:
    labels       - đọc dữ liệu đã tiền xử lý, tạo bảng cặp (người dùng, sản phẩm), nhãn, chia tập theo người dùng
    features     - tạo thuộc tính cho từng cặp, chỉ từ các đơn prior
    discretize   - rời rạc hóa theo entropy (Fayyad và Irani, 1993), chỉ học trên phần huấn luyện
    step4_link   - nối kết quả luật kết hợp của bước 4 thành một thuộc tính cho bước 5
    cba          - CBA (Liu, Hsu và Ma, 1998): CBA-RG và CBA-CB (phương án M1)
    cmar         - CMAR (Li, Han và Pei, 2001): cắt tỉa ba lớp và phân lớp bằng chi bình phương có trọng số
    baselines    - mô hình đối chứng: cây quyết định tiêu chí entropy, LightGBM, luật một thuộc tính
    evaluation   - độ đo, bootstrap theo người dùng, biểu đồ
"""

__version__ = "1.0.0"
