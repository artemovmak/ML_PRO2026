"""Данные рекомендаций: сырая выгрузка, очищенный слой и его схема pandera.

  uv run python -m churn.recs.data      шаг validate: чистка, проверка, отчёт в MLflow; при нарушениях код выхода 1

Сырой слой datasets/online_retail.parquet лежит в DVC и руками не правится. Чистка как в лекции 4.2:
без возвратов (номер чека на C), без строк без клиента, без служебных позиций (код товара не с цифры),
количество и цена больше нуля. Схема проверяет уже очищенный слой: что пришло, то и проверяем.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

RAW_PATH = Path(os.getenv("RECS_DATA_PATH", "datasets/online_retail.parquet"))
EXPERIMENT = os.getenv("RECS_EXPERIMENT", "recs")

# правила для каждой строки: их можно проверить и на маленьком образце в CI
ROWS = pa.DataFrameSchema(
    {
        "InvoiceNo": pa.Column(str, pa.Check.str_matches(r"^\d{6}$")),
        "StockCode": pa.Column(str, pa.Check.str_matches(r"^\d{5}[A-Za-z]{0,2}$")),
        "Description": pa.Column(str, pa.Check.str_length(min_value=1)),
        "Quantity": pa.Column(int, pa.Check.in_range(1, 100_000)),
        "InvoiceDate": pa.Column("datetime64[ns]", pa.Check.in_range(pd.Timestamp("2010-12-01"), pd.Timestamp("2012-01-01"))),
        "UnitPrice": pa.Column(float, pa.Check.in_range(0.001, 1000)),  # фунты, в данных максимум 649.5
        "CustomerID": pa.Column(int),
        "Country": pa.Column(str),
    },
    strict=True,
)

# правила для выгрузки целиком: объём и распределение, на образце не проверить
EXPORT = pa.DataFrameSchema(
    ROWS.columns,
    checks=[
        pa.Check(lambda df: len(df) >= 100_000, error="меньше 100 тысяч строк: выгрузка неполная"),
        pa.Check(lambda df: df["CustomerID"].nunique() >= 1_000, error="меньше 1000 клиентов"),
        pa.Check(lambda df: 0.5 <= df["UnitPrice"].median() <= 10, error="медиана цены вне 0.5..10 фунтов: сменились единицы?"),
    ],
    strict=True,
)


def file_md5(path: Path) -> str:
    """Тот же md5, что DVC пишет в .dvc-файл: по нему видно, на какой версии данных обучена модель."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].str.startswith("C") & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[df["StockCode"].str.match(r"^\d")]
    return df.assign(
        CustomerID=df["CustomerID"].astype("int64"),
        InvoiceDate=pd.to_datetime(df["InvoiceDate"]).astype("datetime64[ns]"),
        Description=df["Description"].fillna("").str.strip(),
    ).reset_index(drop=True)


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    return EXPORT.validate(clean(load_raw(path)), lazy=True)


def main() -> int:
    import mlflow

    raw = load_raw()
    df = clean(raw)
    md5 = file_md5(RAW_PATH)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="validate"):
        mlflow.log_params({"step": "validate", "data": str(RAW_PATH), "data_md5": md5})
        mlflow.log_metrics({"raw_rows": len(raw), "clean_rows": len(df)})
        try:
            EXPORT.validate(df, lazy=True)
        except pa.errors.SchemaErrors as e:
            fc = e.failure_cases
            summary = fc.groupby(["column", "check"], dropna=False).size().rename("cases").reset_index()
            with tempfile.TemporaryDirectory() as tmp:
                fc.head(1000).to_csv(Path(tmp) / "failure_cases.csv", index=False)
                mlflow.log_artifact(str(Path(tmp) / "failure_cases.csv"))
            mlflow.log_metric("failed_checks", len(summary))
            mlflow.set_tag("valid", "false")
            print("данные не прошли проверку, отчёт failure_cases.csv в MLflow:")
            print(summary.to_string(index=False))
            return 1
        mlflow.set_tag("valid", "true")
        mlflow.log_metrics({"clients": df["CustomerID"].nunique(), "items": df["StockCode"].nunique()})
    result = {"valid": True, "data_md5": md5, "clean_rows": len(df), "clients": int(df["CustomerID"].nunique()),
              "items": int(df["StockCode"].nunique()), "last_date": str(df["InvoiceDate"].max())}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
