"""Kiểm thử đơn vị cho mã bước 5. Chạy: python -m pytest tests -q (từ thư mục gốc của repo)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from b5_classification.cba import CBAClassifier, count_condsets, encode_codes, pessimistic_error  # noqa: E402
from b5_classification.cmar import CMARClassifier, chi2_2x2, max_chi2  # noqa: E402
from b5_classification.discretize import Discretizer, mdlp_cut_points  # noqa: E402


def test_cba_paper_example_support_confidence():
    """Ví dụ trong mục 'Basic concepts' của bài CBA: condset {(A,1),(B,1)} xuất hiện 3 lần, 2 lần lớp 1,
    |D| = 10 nên support = 20% và confidence = 66,7%."""
    X = pd.DataFrame({"A": [1, 1, 1, 2, 2, 2, 2, 1, 2, 2], "B": [1, 1, 1, 1, 2, 2, 1, 2, 2, 2]})
    y = np.array([1, 1, 2, 2, 2, 1, 2, 1, 2, 2])
    m = CBAClassifier(min_support=0.10, min_confidence=0.5, max_len=2, prune=False).fit(X, y)
    t = m.table_
    key = ((0, 1 + 1), (1, 1 + 1))          # mã được dịch +1 bên trong
    cnt = t.counts_of(key)
    assert cnt is not None and cnt.sum() == 3
    cars = m.cars_.assign(items=[t.condsets[c] for c in m.cars_["cid"]])
    row = cars[cars["items"] == key].iloc[0]
    assert m.classes_[row["cls_idx"]] == 1
    assert abs(row["support"] - 0.20) < 1e-12
    assert abs(row["confidence"] - 2 / 3) < 1e-12


def test_cba_learns_perfect_rules():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.integers(0, 3, 2000), "b": rng.integers(0, 3, 2000), "noise": rng.integers(0, 4, 2000)})
    y = ((X["a"] == 2) | ((X["a"] == 1) & (X["b"] == 0))).astype(int).to_numpy()
    m = CBAClassifier(min_support=0.01, min_confidence=0.5, max_len=2, prune=True).fit(X, y)
    assert (m.predict(X) == y).mean() == 1.0
    assert m.rules_table().shape[0] >= 2


def test_cba_selection_respects_conditions():
    """Điều kiện 2 của bài báo: mỗi luật được chọn phân lớp đúng ít nhất một dòng khi được chọn."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({c: rng.integers(0, 3, 3000) for c in "abcd"})
    y = (rng.random(3000) < 0.2 + 0.5 * (X["a"] == 2)).astype(int)
    m = CBAClassifier(min_support=0.01, min_confidence=0.5, max_len=3).fit(X, y)
    assert all(s["train_correct"] >= 1 for s in m.rules_)
    tr = m.selection_trace_
    if len(tr):
        cut = int(np.argmin(tr["total_errors"].to_numpy()))
        assert len(m.rules_) == cut + 1


def test_pessimistic_error_monotone():
    e = pessimistic_error(np.array([0, 1, 5]), np.array([10, 10, 10]))
    assert np.all(np.diff(e) > 0) and np.all(e <= 1)


def test_mdlp_finds_true_cut():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 100, 5000)
    y = (x > 60).astype(int)
    cuts = mdlp_cut_points(x, y, max_intervals=5)
    assert len(cuts) == 1 and abs(cuts[0] - 60) < 1
    noise = rng.integers(0, 2, 5000)
    assert mdlp_cut_points(rng.uniform(0, 1, 5000), noise, None) == []


def test_discretizer_nan_and_cap():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < x).astype(int)          # quan hệ đơn điệu: MDLP gốc sẽ cắt nhiều lần
    df = pd.DataFrame({"x": x})
    df.loc[:9, "x"] = np.nan
    d = Discretizer(max_intervals=4).fit(df, ["x"], y)
    codes = d.transform(df)
    assert len(d.cuts["x"]) <= 3
    assert (codes.loc[:9, "x"] == -1).all()
    assert len(Discretizer(max_intervals=None).fit(df, ["x"], y).cuts["x"]) > 3


def test_chi2_bounded_by_max_chi2():
    rng = np.random.default_rng(4)
    n = 1000
    sp = rng.integers(10, 500, 200)
    sc = rng.integers(10, 500, 200)
    spc = np.array([rng.integers(max(0, a + b - n), min(a, b) + 1) for a, b in zip(sp, sc)])
    chi = chi2_2x2(sp, sc, spc, n)
    mx = max_chi2(sp, sc, n)
    assert np.all(chi <= mx + 1e-6)


def test_cmar_runs_and_beats_majority():
    rng = np.random.default_rng(5)
    X = pd.DataFrame({c: rng.integers(0, 3, 4000) for c in "abc"})
    y = (rng.random(4000) < 0.1 + 0.7 * ((X["a"] == 2) & (X["b"] != 0))).astype(int).to_numpy()
    m = CMARClassifier(min_support=0.01, min_confidence=0.5, max_len=2).fit(X, y)
    acc = (m.predict(X) == y).mean()
    assert acc > max(y.mean(), 1 - y.mean())
    s = m.predict_score(X)
    assert np.all((s >= 0) & (s <= 1))
