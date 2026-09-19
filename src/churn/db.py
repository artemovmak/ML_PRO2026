import psycopg
from psycopg.types.json import Json

from churn.config import settings

DDL = """
CREATE TABLE IF NOT EXISTS predictions (

    request_id      uuid PRIMARY KEY,
    ts      timestamptz NOT NULL DEFAULT now(),
    model_version       text NOT NULL,
    features        jsonb NOT NULL,
    score       double precision NOT NULL,
    latency_ms real
)
"""

def init() -> None:
    if not settings.database_url:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(7001)")
        conn.execute(DDL)


def save_prediction(request_id : str, features : dict, score : float, model_version : str, latency_ms : float) -> None:
    if not settings.database_url:
        return
    with psycopg.connect(settings.database_url) as conn:
            conn.execute(
            "INSERT INTO predictions (request_id, model_version, features, score, latency_ms) "
            "VALUES (%s, %s, %s, %s, %s)",
            (request_id, model_version, Json(features), score, latency_ms),
        )

    