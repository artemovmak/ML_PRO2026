# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["openai", "implicit", "pandas", "pyarrow", "scipy"]
# ///
"""Реранкер в проде, как на слайдах 15 и 16 лекции 5.3: ALS даёт 20 кандидатов, vLLM переставляет, при ошибке порядок ALS.

Запуск с ноутбука через SSH-туннель на арендованную машину (bash laptop/tunnel.sh), из папки week5;
зависимости uv берёт из шапки файла:

  uv run client/rerank_demo.py [--timeout 0.15] [--customers 5] [--model reranker|base]

Что показывает: ответ модели для нескольких клиентов, время каждого запроса, долю запасного пути.
Скор кандидата = logprob его буквы на позиции ответа (logprob_token_ids), генерации нет.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit иначе ругается на пул потоков OpenBLAS
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CUT, END, LETTERS, SEED, als_candidates, item_names, load_retail, ndcg_at10, prompt  # noqa: E402

LETTER_IDS = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # токены A..T у Qwen2.5


def rerank(client, messages, model):
    """Один прямой проход, 20 logprobs. Любая ошибка (таймаут, 5xx) наверх: решает вызывающий."""
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=1, temperature=0, logprobs=True,
        extra_body={"logprob_token_ids": LETTER_IDS})
    lp = {t.token: t.logprob for t in resp.choices[0].logprobs.content[0].top_logprobs}
    return np.array([lp.get(L, -1e9) for L in LETTERS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="reranker")
    ap.add_argument("--timeout", type=float, default=0.15, help="бюджет на реранкер, секунды")
    ap.add_argument("--customers", type=int, default=5)
    a = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=a.url, api_key="EMPTY", timeout=a.timeout, max_retries=0)

    df = load_retail()
    names = item_names(df)
    rows = [r for r in als_candidates(df, CUT, END) if r["truth"] & set(r["cands"])]
    rng = np.random.default_rng(SEED)
    rows = [rows[i] for i in rng.choice(len(rows), a.customers, replace=False)]

    # Прогрев, как у сервиса при старте: открыть соединение и подгрузить адаптер с запасом по времени.
    # Без него первый запрос через туннель не влезает в бюджет, клиент рвёт соединение по таймауту,
    # и каждый следующий снова платит за новое соединение: запасной путь у всех подряд.
    try:
        client.with_options(timeout=5).chat.completions.create(
            model=a.model, messages=[{"role": "user", "content": "A"}], max_tokens=1)
    except Exception:  # noqa: BLE001  сервер лежит: это покажет основной цикл
        pass

    fallback, lat = 0, []
    for r in rows:
        t = time.perf_counter()
        try:
            scores = rerank(client, prompt(r, names), a.model)
            order = [r["cands"][j] for j in np.argsort(-scores, kind="stable")]
            source = a.model
        except Exception as e:  # noqa: BLE001  таймаут, 5xx, соединение: порядок ALS
            order, source, fallback = list(r["cands"]), f"als (fallback: {type(e).__name__})", fallback + 1
        lat.append(time.perf_counter() - t)
        bought = r["truth"] & set(r["cands"])
        print(f"\nклиент {r['user']}: {lat[-1] * 1000:.0f} мс, источник {source}, "
              f"NDCG@10 {ndcg_at10(order, r['truth']):.3f} (порядок ALS {ndcg_at10(r['cands'], r['truth']):.3f})")
        for k, s in enumerate(order[:5], 1):
            print(f"  {k}. {names[s]}{'   <- купил' if s in bought else ''}")
    print(f"\nзапросов {len(rows)}, медиана {np.median(lat) * 1000:.0f} мс, запасной путь {fallback}/{len(rows)}")


if __name__ == "__main__":
    main()
