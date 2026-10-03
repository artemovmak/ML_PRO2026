"""Модель рекомендаций в реестре MLflow: обёртка pyfunc над факторами ALS.

На вход таблица с колонкой customer_id, на выход по k строк на клиента: rank, stock_code, description, score,
source. Знакомым клиентам ALS без уже купленного, незнакомым популярное (source = popular).
"""
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ARTIFACTS = ["user_factors.npy", "item_factors.npy", "seen.npz", "users.json", "items.json", "popular.json"]


class RecsModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        a = {name: Path(path) for name, path in context.artifacts.items()}
        self.user_factors = np.load(a["user_factors.npy"])
        self.item_factors = np.load(a["item_factors.npy"])
        self.seen = load_npz(a["seen.npz"]).tocsr()
        self.users = {u: i for i, u in enumerate(json.loads(a["users.json"].read_text()))}
        self.items = json.loads(a["items.json"].read_text(encoding="utf-8"))  # [[stock_code, description], ...]
        self.popular = json.loads(a["popular.json"].read_text())

    def recommend(self, customer_id: int, k: int = 10) -> tuple[list[int], list[float], str]:
        u = self.users.get(int(customer_id))
        if u is None:
            return self.popular[:k], [0.0] * k, "popular"
        scores = self.item_factors @ self.user_factors[u]
        scores[self.seen[u].indices] = -np.inf  # купленное не советуем
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return top.tolist(), scores[top].tolist(), "als"

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        k = int((params or {}).get("k", 10))
        rows = []
        for cid in model_input["customer_id"]:
            idx, scores, source = self.recommend(cid, k)
            rows += [(int(cid), rank, *self.items[i], round(float(s), 4), source)
                     for rank, (i, s) in enumerate(zip(idx, scores, strict=False), start=1)]
        return pd.DataFrame(rows, columns=["customer_id", "rank", "stock_code", "description", "score", "source"])
