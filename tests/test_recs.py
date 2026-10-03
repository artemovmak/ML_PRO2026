"""Рекомендации без данных и без базы: в CI датасета нет, он в DVC. Схема проверяется на маленьком образце."""
import pandas as pd
import pandera.pandas as pa
import pytest

from churn import db
from churn.recs.data import EXPORT, ROWS, clean
from churn.recs.metrics import ranking_metrics


@pytest.fixture()
def raw():
    """Пять строк сырой выгрузки: обычная покупка, возврат, строка без клиента, служебная позиция, нулевая цена."""
    return pd.DataFrame({
        "InvoiceNo": ["536365", "C536379", "536414", "536370", "536381"],
        "StockCode": ["85123A", "22423", "22139", "POST", "22728"],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER", "REGENCY CAKESTAND 3 TIER", None, "POSTAGE", "ALARM CLOCK BAKELIKE PINK"],
        "Quantity": [6, -1, 56, 3, 24],
        "InvoiceDate": pd.to_datetime(["2010-12-01 08:26", "2010-12-01 09:41", "2010-12-01 11:52", "2010-12-01 08:45", "2010-12-01 09:45"]),
        "UnitPrice": [2.55, 12.75, 0.0, 18.0, 3.75],
        "CustomerID": [17850.0, 14527.0, None, 12583.0, 15311.0],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "France", "United Kingdom"],
    })


def test_clean_keeps_only_purchases(raw):
    df = clean(raw)
    assert df["InvoiceNo"].tolist() == ["536365", "536381"]
    assert df["CustomerID"].dtype == "int64"
    ROWS.validate(df)


def test_price_in_pence_is_caught(raw):
    bad = clean(raw).assign(UnitPrice=lambda d: d["UnitPrice"] * 100 + 1000)
    with pytest.raises(pa.errors.SchemaErrors) as e:
        ROWS.validate(bad, lazy=True)
    assert set(e.value.failure_cases["column"]) == {"UnitPrice"}


def test_small_export_is_rejected(raw):
    with pytest.raises(pa.errors.SchemaErrors) as e:
        EXPORT.validate(clean(raw), lazy=True)
    assert "меньше 100 тысяч строк: выгрузка неполная" in set(e.value.failure_cases["check"])


def test_metrics_lecture_example():
    """Пример из лекции 4.3: список A..E, потом куплены B, D и F."""
    m = ranking_metrics({0: ["A", "B", "C", "D", "E"]}, {0: {"B", "D", "F"}}, k=5, n_items=10)
    assert m["precision"] == pytest.approx(0.40)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["hitrate"] == 1
    assert m["ndcg"] == pytest.approx(0.4982, abs=1e-3)


def test_recommend_known_customer(client, monkeypatch):
    monkeypatch.setattr(db, "fetch_recommendations", lambda cid, k: ([("22423", "regency cakestand 3 tier")] * k, "als", "recs-v2"))
    r = client.get("/v1/recommend/12347?k=3")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "als" and len(body["items"]) == 3 and body["model_version"] == "recs-v2"


def test_recommend_not_published_is_503(client, monkeypatch):
    monkeypatch.setattr(db, "fetch_recommendations", lambda cid, k: None)
    assert client.get("/v1/recommend/12347").status_code == 503


def test_recommend_k_out_of_range_is_422(client):
    assert client.get("/v1/recommend/12347?k=50").status_code == 422
