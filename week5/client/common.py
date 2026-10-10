# -*- coding: utf-8 -*-
"""Общее для экспериментов недели 5: данные, протокол недели 4, кандидаты ALS, промпт реранкера, метрики, окружение.

Протокол (не менять): срез 1.09.2011, до него обучение, после проверка до конца данных; релевантны только новые
для клиента товары; купленное до среза из рекомендаций убирается. Данные: Online Retail, UCI, CC BY 4.0,
doi 10.24432/C5BW33, очищены как в неделе 4 (data/retail_clean.parquet).
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit просит, на результат ALS не влияет (проверено)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "retail_clean.parquet"
DATA_SHA256 = "d40ae9a902ba8b88a1c9f9d9b441af4ace897c136790c08e2dbfa0ac37b72d5f"
RESULTS = ROOT / "results"
SEED = 42
CUT = pd.Timestamp("2011-09-01")
END = pd.Timestamp("2012-01-01")
WINDOWS_TRAIN = [("2011-05-01", "2011-07-01"), ("2011-07-01", "2011-09-01")]
N_CAND, N_HIST = 20, 10
LETTERS = [chr(ord("A") + i) for i in range(N_CAND)]
SYSTEM = "You rank products for an online gift shop."
QUESTION = "Which candidate will the customer buy next? Answer with one letter."
MODELS = {
    "0.5B": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "1.5B": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "7B": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
ALS_PARAMS = dict(factors=64, regularization=0.1, alpha=10.0, iterations=20, random_state=42)


# ---------------------------------------------------------------- данные
def load_retail():
    import hashlib
    h = hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert h == DATA_SHA256, f"data/retail_clean.parquet изменён: {h}"
    df = pd.read_parquet(DATA)
    df["d"] = pd.to_datetime(df.InvoiceDate)
    return df


def interactions(df):
    """Пара клиент-товар, вес = число разных заказов с этим товаром."""
    return df.groupby(["CustomerID", "StockCode"]).InvoiceNo.nunique().rename("n").reset_index()


def item_names(df):
    """Самое частое описание товара, строчными: так короче в токенах."""
    names = (df.assign(desc=df.Description.fillna("").str.strip().str.lower())
               .groupby("StockCode").desc.agg(lambda s: s.mode().iat[0] if len(s.mode()) else ""))
    return {k: (v if v else k) for k, v in names.items()}


def to_csr(tr, users, items):
    from scipy.sparse import csr_matrix
    return csr_matrix((np.log1p(tr.n.to_numpy()).astype(np.float32),
                       (tr.CustomerID.map(users).to_numpy(), tr.StockCode.map(items).to_numpy())),
                      shape=(len(users), len(items)))


def split(df, cut=CUT, end=END):
    """Обучающие пары до cut; правда: новые для клиента товары в [cut, end)."""
    hist, fut = df[df.d < cut], df[(df.d >= cut) & (df.d < end)]
    tr, te = interactions(hist), interactions(fut)
    users = {u: i for i, u in enumerate(sorted(tr.CustomerID.unique()))}
    items = {s: i for i, s in enumerate(sorted(tr.StockCode.unique()))}
    te = te[te.CustomerID.isin(users) & te.StockCode.isin(items)]
    seen = tr.groupby("CustomerID").StockCode.apply(set)
    te = te[[s not in seen[c] for c, s in zip(te.CustomerID, te.StockCode)]]
    truth = {c: set(g.StockCode) for c, g in te.groupby("CustomerID")}
    return hist, tr, users, items, truth


def fit_als(ui):
    """ALS из недели 4. Всегда CPU-реализация implicit, чтобы кандидаты совпадали с эталоном."""
    from implicit.cpu.als import AlternatingLeastSquares
    als = AlternatingLeastSquares(**ALS_PARAMS)
    als.fit(ui, show_progress=False)
    return als


def als_candidates(df, cut, end):
    """ALS на данных до cut. Для клиентов с новыми покупками в [cut, end): 20 кандидатов, правда, история."""
    hist, tr, users, items, truth = split(df, cut, end)
    inv = {i: s for s, i in items.items()}
    ui = to_csr(tr, users, items)
    als = fit_als(ui)
    cs = sorted(truth)
    ids, _ = als.recommend(np.array([users[c] for c in cs]), ui[[users[c] for c in cs]], N=N_CAND,
                           filter_already_liked_items=True)
    recent = hist.sort_values(["d", "InvoiceNo"]).drop_duplicates(["CustomerID", "StockCode"], keep="last")
    hist_items = recent.groupby("CustomerID").StockCode.apply(lambda s: list(s)[-N_HIST:])
    return [{"user": int(c), "cands": [inv[i] for i in row], "truth": truth[c], "history": hist_items[c]}
            for c, row in zip(cs, ids)]


# ---------------------------------------------------------------- промпт реранкера (один для обучения и сервинга)
def prompt(r, names):
    hist = "\n".join(f"- {names[s]}" for s in r["history"])
    cands = "\n".join(f"{L}. {names[s]}" for L, s in zip(LETTERS, r["cands"]))
    text = f"The customer bought recently (oldest to newest):\n{hist}\n\nCandidates:\n{cands}\n\n{QUESTION}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


# ---------------------------------------------------------------- метрики
def ndcg_at10(order, rel):
    g = [1 if s in rel else 0 for s in order[:10]]
    dcg = sum(x / np.log2(j + 2) for j, x in enumerate(g))
    idcg = sum(1 / np.log2(j + 2) for j in range(min(len(rel), 10)))
    return dcg / idcg


def rerank_metrics(rows, scores, n_all):
    """NDCG@10 по n_all клиентам: у кого нет попаданий в 20 кандидатах, у того 0 при любом порядке.
    rows: только клиенты с попаданием; scores: по 20 чисел на клиента, больше = выше."""
    nd, top1 = [], []
    for r, s in zip(rows, scores):
        order = [r["cands"][j] for j in np.argsort(-np.asarray(s, dtype=float), kind="stable")]
        nd.append(ndcg_at10(order, r["truth"]))
        top1.append(int(np.argmax(s)))
    return float(np.sum(nd)) / n_all, np.array(nd), top1


def bootstrap_diff(nd_a, nd_b, n_all, rng, n_boot=2000):
    """Среднее и 95% интервал разницы NDCG@10 (a минус b) по клиентам, бутстреп."""
    diff = np.concatenate([np.asarray(nd_a) - np.asarray(nd_b), np.zeros(n_all - len(nd_a))])
    boots = [rng.choice(diff, size=len(diff)).mean() for _ in range(n_boot)]
    return {"mean": round(float(diff.mean()), 4),
            "ci95": [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)]}


# ---------------------------------------------------------------- окружение и вывод
def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "trl", "datasets", "implicit", "vllm", "numpy", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["cpu_threads"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        pass
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                             "--format=csv,noheader"], capture_output=True, text=True,
                                            timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        info["nvidia_smi"] = None
    return info


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)),
                    encoding="utf-8")
    return path
