"""Рекомендации: обучение по протоколу лекции 4.3, бейзлайны в том же прогоне, гейт и реестр MLflow.

  MLFLOW_TRACKING_URI=http://mlflow.localhost uv run python -m churn.recs.train
  SPLIT=random ...     то же на случайном разбиении пар: только посмотреть на утечку, в реестр не попадает

Протокол: учимся на покупках до CUTOFF, проверяем на товарах, которые клиент впервые взял после него,
купленное до среза из рекомендаций убираем, метрики @10 по клиентам с историей до среза и покупками после.
Популярное, item-kNN и ALS считаются в одном прогоне на одном срезе.
Гейт: разбиение по времени; NDCG@10 у ALS не ниже 1.2 популярного; покрытие топ-10 не ниже 10%;
NDCG@10 выше, чем у текущего champion. Для публикации ALS переобучается на всех данных с теми же параметрами.
"""
import json
import os
import tempfile
from pathlib import Path

import implicit
import mlflow
import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from scipy.sparse import csr_matrix, save_npz

from churn.recs.data import RAW_PATH, file_md5, load_clean
from churn.recs.metrics import ranking_metrics
from churn.recs.model import ARTIFACTS, RecsModel

MODEL_NAME = os.getenv("RECS_MODEL_NAME", "recs")
EXPERIMENT = os.getenv("RECS_EXPERIMENT", "recs")
SPLIT = os.getenv("SPLIT", "time")
CUTOFF = pd.Timestamp(os.getenv("CUTOFF", "2011-09-01"))
FACTORS = int(os.getenv("FACTORS", "64"))
REG = float(os.getenv("REG", "0.1"))
ALPHA = float(os.getenv("ALPHA", "10"))
ITERATIONS = int(os.getenv("ITERATIONS", "20"))
MIN_GAIN = float(os.getenv("GATE_MIN_GAIN", "0.0"))
POP_RATIO, MIN_COVERAGE, K, SEED = 1.2, 0.10, 10, 42
SIGNATURE_OUT = pd.DataFrame({"customer_id": [12347], "rank": [1], "stock_code": ["22423"], "description": ["regency cakestand 3 tier"],
                              "score": [0.5], "source": ["als"]})


def interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Пара клиент-товар, вес = число разных заказов с этим товаром."""
    return df.groupby(["CustomerID", "StockCode"])["InvoiceNo"].nunique().rename("n").reset_index()


def to_csr(pairs: pd.DataFrame, users: dict, items: dict) -> csr_matrix:
    return csr_matrix((np.log1p(pairs["n"].to_numpy()).astype(np.float32),
                       (pairs["CustomerID"].map(users).to_numpy(), pairs["StockCode"].map(items).to_numpy())),
                      shape=(len(users), len(items)))


def fit_als(ui: csr_matrix) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(factors=FACTORS, regularization=REG, alpha=ALPHA, iterations=ITERATIONS, random_state=SEED)
    model.fit(ui, show_progress=False)
    return model


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if SPLIT == "time":
        return interactions(df[df["InvoiceDate"] < CUTOFF]), interactions(df[df["InvoiceDate"] >= CUTOFF])
    pairs = interactions(df).sample(frac=1, random_state=SEED)  # утечка: будущее клиента попадает в обучение
    cut = int(0.8 * len(pairs))
    return pairs.iloc[:cut], pairs.iloc[cut:]


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, int]:
    users = {u: i for i, u in enumerate(sorted(train["CustomerID"].unique()))}
    items = {s: i for i, s in enumerate(sorted(train["StockCode"].unique()))}
    ui = to_csr(train, users, items)
    test = test[test["CustomerID"].isin(users) & test["StockCode"].isin(items)]
    seen = train.groupby("CustomerID")["StockCode"].apply(set)
    test = test[[s not in seen[c] for c, s in zip(test["CustomerID"], test["StockCode"], strict=False)]]  # только новые для клиента
    truth = {users[c]: {items[s] for s in g["StockCode"]} for c, g in test.groupby("CustomerID")}
    uids = np.array(sorted(truth))

    pop_order = np.argsort(-np.asarray((ui > 0).sum(axis=0)).ravel())
    recs = {"popular": {}}
    for u in uids:
        bought = set(ui[u].indices)
        recs["popular"][u] = [i for i in pop_order[:K + len(bought)] if i not in bought][:K]
    knn = CosineRecommender(K=100)
    knn.fit(ui, show_progress=False)
    recs["knn"] = dict(zip(uids, knn.recommend(uids, ui[uids], N=K, filter_already_liked_items=True)[0], strict=False))
    recs["als"] = dict(zip(uids, fit_als(ui).recommend(uids, ui[uids], N=K, filter_already_liked_items=True)[0], strict=False))
    return {name: ranking_metrics(r, truth, K, len(items)) for name, r in recs.items()}, len(uids)


def save_model(df: pd.DataFrame, path: Path) -> None:
    """ALS на всех данных: факторы, словари клиентов и товаров, купленное, популярное. Всё, что нужно RecsModel."""
    pairs = interactions(df)
    users = {u: i for i, u in enumerate(sorted(pairs["CustomerID"].unique()))}
    items = {s: i for i, s in enumerate(sorted(pairs["StockCode"].unique()))}
    ui = to_csr(pairs, users, items)
    model = fit_als(ui)
    names = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0])
    np.save(path / "user_factors.npy", np.asarray(model.user_factors))
    np.save(path / "item_factors.npy", np.asarray(model.item_factors))
    save_npz(path / "seen.npz", ui)
    (path / "users.json").write_text(json.dumps([int(u) for u in users]))
    (path / "items.json").write_text(json.dumps([[s, names[s]] for s in items], ensure_ascii=False), encoding="utf-8")
    popular = np.argsort(-np.asarray((ui > 0).sum(axis=0)).ravel())[:50]
    (path / "popular.json").write_text(json.dumps([int(i) for i in popular]))


def champion_ndcg(client: MlflowClient) -> tuple[str | None, float | None]:
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None, None
    return mv.version, client.get_run(mv.run_id).data.metrics.get("ndcg_10")


def main() -> dict:
    df = load_clean(RAW_PATH)
    train, test = split(df)
    scores, n_users = evaluate(train, test)
    als, pop = scores["als"], scores["popular"]

    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    with mlflow.start_run(run_name=f"als-{SPLIT}") as run:
        mlflow.log_params({"split": SPLIT, "cutoff": str(CUTOFF.date()) if SPLIT == "time" else "-", "factors": FACTORS,
                           "regularization": REG, "alpha": ALPHA, "iterations": ITERATIONS, "k": K,
                           "data": str(RAW_PATH), "data_md5": file_md5(RAW_PATH), "implicit": implicit.__version__})
        mlflow.log_metric("eval_users", n_users)
        for name, m in scores.items():
            mlflow.log_metrics({f"{name}_{metric}_10": round(v, 4) for metric, v in m.items()})
        mlflow.log_metrics({"ndcg_10": als["ndcg"], "coverage_10": als["coverage"]})
        version = None
        if SPLIT == "time":  # модель со случайного разбиения в реестр не попадает вовсе
            with tempfile.TemporaryDirectory() as tmp:
                save_model(df, Path(tmp))
                info = mlflow.pyfunc.log_model(
                    name="model", python_model=RecsModel(), registered_model_name=MODEL_NAME,
                    artifacts={name: str(Path(tmp) / name) for name in ARTIFACTS},
                    signature=infer_signature(pd.DataFrame({"customer_id": [12347]}), SIGNATURE_OUT, params={"k": K}),
                    pip_requirements=[f"mlflow-skinny=={mlflow.__version__}", "numpy", "pandas", "scipy"])
            version = info.registered_model_version

    old_version, old_ndcg = champion_ndcg(client) if version else (None, None)
    checks = {"time_split": SPLIT == "time", "beats_popular": als["ndcg"] >= POP_RATIO * pop["ndcg"],
              "coverage": als["coverage"] >= MIN_COVERAGE,
              "beats_champion": old_ndcg is None or als["ndcg"] > old_ndcg + MIN_GAIN}
    promoted = all(checks.values())
    if version:
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        if promoted:
            client.set_registered_model_alias(MODEL_NAME, "champion", version)

    result = {"run_id": run.info.run_id, "split": SPLIT, "version": version,
              "ndcg_10": {name: round(m["ndcg"], 4) for name, m in scores.items()},
              "coverage_10": round(als["coverage"], 4), "champion_before": old_version, "checks": checks, "promoted": promoted}
    print(json.dumps(result, ensure_ascii=False))
    xcom = Path("/airflow/xcom")
    if xcom.is_dir():
        (xcom / "return.json").write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
