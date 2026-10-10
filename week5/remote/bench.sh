#!/usr/bin/env bash
# Нагрузочный тест в форме запроса реранкера: 300 токенов на вход, 1 на выход, 40 запросов в секунду.
#
#   bash bench.sh uv [rate]      bash bench.sh docker [rate]
set -euo pipefail
MODE=${1:-uv}
RATE=${2:-40}
CMD="vllm bench serve --backend openai-chat --base-url http://localhost:8000 --endpoint /v1/chat/completions \
  --model Qwen/Qwen2.5-1.5B-Instruct --served-model-name reranker \
  --dataset-name random --random-input-len 300 --random-output-len 1 \
  --num-prompts 500 --request-rate $RATE --percentile-metrics ttft,e2el --metric-percentiles 50,95,99"
case $MODE in
  uv)     PATH=/opt/vllm/bin:$PATH $CMD ;;
  docker) docker exec vllm $CMD ;;
esac
