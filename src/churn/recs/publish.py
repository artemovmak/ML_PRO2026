"""Публикация: топ-10 от champion для всех клиентов в Postgres, плюс популярное для незнакомых.

  DATABASE_URL=postgresql://... MLFLOW_TRACKING_URI=... uv run python -m churn.recs.publish

Пакетный режим из лекции 4.2: списки считаются заранее, сервис только читает по ключу.
Таблицы собираются заново рядом (*_new) и подменяются одной транзакцией: сервис никогда не видит
половину таблицы, а если публикация упала, остаются вчерашние списки.
"""
import json
import os
import time
from pathlib import Path

import mlflow
import pandas as pd
import psycopg
from mlflow import MlflowClient

MODEL_NAME = os.getenv("RECS_MODEL_NAME", "recs")
ALIAS = os.getenv("RECS_MODEL_ALIAS", "champion")
K = 10

DDL = """
CREATE TABLE recommendations_new (customer_id bigint NOT NULL, rank smallint NOT NULL, stock_code text NOT NULL,
    description text NOT NULL, score real NOT NULL, model_version text NOT NULL, PRIMARY KEY (customer_id, rank));
CREATE TABLE recs_popular_new (rank smallint PRIMARY KEY, stock_code text NOT NULL, description text NOT NULL,
    model_version text NOT NULL);
"""
SWAP = """
DROP TABLE IF EXISTS recommendations; ALTER TABLE recommendations_new RENAME TO recommendations;
ALTER INDEX recommendations_new_pkey RENAME TO recommendations_pkey;
DROP TABLE IF EXISTS recs_popular; ALTER TABLE recs_popular_new RENAME TO recs_popular;
ALTER INDEX recs_popular_new_pkey RENAME TO recs_popular_pkey;
"""


def main() -> dict:
    t0 = time.perf_counter()
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
    version = f"{MODEL_NAME}-v{mv.version}"
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    inner = model.unwrap_python_model()
    customers = pd.DataFrame({"customer_id": list(inner.users)})
    recs = model.predict(customers, params={"k": K})
    popular = model.predict(pd.DataFrame({"customer_id": [-1]}), params={"k": K})  # незнакомый клиент

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(7002)")  # две публикации сразу не пойдут
        conn.execute("DROP TABLE IF EXISTS recommendations_new; DROP TABLE IF EXISTS recs_popular_new")
        conn.execute(DDL)
        with conn.cursor().copy("COPY recommendations_new FROM STDIN") as copy:
            for r in recs.itertuples(index=False):
                copy.write_row((r.customer_id, r.rank, r.stock_code, r.description, r.score, version))
        with conn.cursor().copy("COPY recs_popular_new FROM STDIN") as copy:
            for r in popular.itertuples(index=False):
                copy.write_row((r.rank, r.stock_code, r.description, version))
        conn.execute(SWAP)
    result = {"model_version": version, "customers": len(customers), "rows": len(recs),
              "seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps(result, ensure_ascii=False))
    xcom = Path("/airflow/xcom")
    if xcom.is_dir():
        (xcom / "return.json").write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
