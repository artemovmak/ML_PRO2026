#!/usr/bin/env bash
# vLLM с базой Qwen2.5-1.5B и адаптерами реранкера из недели 5. Слушает только 127.0.0.1: наружу через SSH-туннель.
#
#   bash serve.sh uv        из /opt/vllm
#   bash serve.sh docker    из образа vllm/vllm-openai, та же команда внутри контейнера
set -euo pipefail
MODE=${1:-uv}
MODEL=Qwen/Qwen2.5-1.5B-Instruct
REV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
ARGS="$MODEL --revision $REV --host 0.0.0.0 --port 8000 --served-model-name base \
  --max-model-len 2048 --gpu-memory-utilization 0.90 --max-num-seqs 256 \
  --enable-lora --max-lora-rank 16 --max-loras 4 \
  --lora-modules reranker=ADAPTERS/reranker-1.5B"

case $MODE in
  uv)
    export PATH=/opt/vllm/bin:$PATH VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
    exec vllm serve ${ARGS//ADAPTERS/$HOME/adapters} --host 127.0.0.1
    ;;
  docker)
    exec docker run --rm --name vllm --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/adapters:/adapters" \
      -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
      vllm/vllm-openai:v0.30.0 ${ARGS//ADAPTERS//adapters}
    ;;
esac
