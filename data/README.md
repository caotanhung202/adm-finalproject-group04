# Dữ liệu

Thư mục này không chứa dữ liệu gốc. Khi chạy lại, đặt tệp như sau:

```
data/
├── raw/                                   6 tệp CSV tải từ Kaggle (không đưa lên GitHub)
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── products.csv
│   ├── aisles.csv
│   └── departments.csv
└── processed/
    ├── orders_prior_sample.parquet          5 tệp do 02_preprocessing.ipynb tạo ra
    ├── orders_train_sample.parquet          (không đưa lên GitHub)
    ├── order_products_prior_sample.parquet
    ├── order_products_train_sample.parquet
    ├── products_full.parquet
    ├── rules_aisle.csv                      luật gian hàng của bước 4 (có trong repo)
    └── b5/                                  kết quả bước 5 (có trong repo)
```

Kiểm tra nhanh sau bước 2: `order_products_prior_sample.parquet` phải có 2.066.000 dòng thuộc 206.657 đơn hàng, `orders_train_sample.parquet` có 13.121 đơn, và mọi đơn prior đều có ít nhất một dòng sản phẩm.
