"""Скачать Online Retail и сохранить сырую выгрузку как есть: datasets/online_retail.parquet.

Данные: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository, CC BY 4.0.
https://doi.org/10.24432/C5BW33. Интернет-магазин подарков, все чеки с 1.12.2010 по 9.12.2011.

  uv run --with openpyxl python scripts/get_online_retail.py

Сырой слой ничем не чистится: возвраты, строки без клиента и служебные позиции остаются.
Чистка и проверка живут в churn.recs.data, это следующий слой.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
OUT = Path("datasets/online_retail.parquet")

raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=120).read()))
df = pd.read_excel(raw.open("Online Retail.xlsx"), dtype={"InvoiceNo": str, "StockCode": str, "Description": str})
OUT.parent.mkdir(exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"{OUT}: {len(df)} строк, {df.InvoiceDate.min()} .. {df.InvoiceDate.max()}, {OUT.stat().st_size / 1e6:.1f} МБ")
