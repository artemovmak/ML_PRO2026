"""Испорченная выгрузка для проверки шага validate: источник тихо перешёл с фунтов на пенсы.

  uv run python scripts/break_export.py

Перезаписывает datasets/online_retail.parquet, ни одной ошибки при этом не возникает: файл читается,
типы те же, строк столько же. Ловит такое только проверка значений и распределения.
Вернуть как было: git checkout -- datasets/online_retail.parquet.dvc && uv run dvc checkout
"""
from pathlib import Path

import pandas as pd

PATH = Path("datasets/online_retail.parquet")
df = pd.read_parquet(PATH)
df["UnitPrice"] = df["UnitPrice"] * 100
df.to_parquet(PATH, index=False)
print(f"{PATH}: цены в пенсах, медиана {df['UnitPrice'].median():.0f}, строк {len(df)}")
