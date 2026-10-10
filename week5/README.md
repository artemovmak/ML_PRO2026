# Семинар 5 · vLLM на арендованной GPU, с нуля за час

Маршрут преподавателя: `walkthrough05.pdf`, все команды по порядку: `commands.md`. Папка самодостаточна, ничего из других недель не нужно. Команды запускать из Git Bash, из этой папки.

| Что | Где |
|---|---|
| адрес машины и ключ | `laptop/machine.env` |
| залить и поставить всё на свежую машину (~8 мин) | `bash laptop/push.sh` |
| шелл на машине плюс туннель на `localhost:8000` | `bash laptop/tunnel.sh` |
| на машине: сервер, нагрузка | `remote/serve.sh`, `remote/bench.sh` (копируются в `~/seminar`) |
| адаптер реранкера из недели 5 (LoRA r=16 на Qwen2.5-1.5B) | `adapters/reranker-1.5B` |
| тела запросов для curl | `client/chat.json`, `client/lora_v2.json` |
| тот же запрос из Python | `uv run client/hello.py` |
| клиент с таймаутом и запасным путём, данные Online Retail | `client/rerank_demo.py`, `client/common.py`, `client/data` |
| манифест для кластера с GPU, на семинаре не используется | `k8s/vllm.yaml` |

Клиент с ноутбука, зависимости `uv` берёт из шапки скрипта:

```
uv run client/rerank_demo.py --customers 5
```
