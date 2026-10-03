"""Метрики top-K из лекции 4.3. На вход: списки рекомендаций и множества того, что клиент взял потом."""
import numpy as np


def ranking_metrics(recs: dict, truth: dict, k: int, n_items: int) -> dict:
    """recs: клиент -> список товаров (не короче k); truth: клиент -> множество товаров из теста."""
    precision, recall, ndcg, hit, shown = [], [], [], [], set()
    for user, rel in truth.items():
        top = list(recs[user])[:k]
        gains = [1 if item in rel else 0 for item in top]
        hits = sum(gains)
        precision.append(hits / k)
        recall.append(hits / len(rel))
        hit.append(1 if hits else 0)
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
        ndcg.append(dcg / idcg)
        shown.update(top)
    return {"precision": float(np.mean(precision)), "recall": float(np.mean(recall)), "ndcg": float(np.mean(ndcg)),
            "hitrate": float(np.mean(hit)), "coverage": len(shown) / n_items}
