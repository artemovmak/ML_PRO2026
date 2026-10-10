#!/usr/bin/env bash
# Подготовка арендованной GPU-машины к семинару 5: vLLM и веса. Запускать от root, повторный запуск безопасен.
#
#   bash setup.sh docker    образ vllm/vllm-openai (нужны Docker и NVIDIA Container Toolkit), тот же, что в k8s/vllm.yaml
#   bash setup.sh uv        vLLM как pip-пакет в /opt/vllm, если Docker на машине нет
#
# Веса кладутся в ~/.cache/huggingface, адаптеры ждём в ~/adapters (scp с ноутбука).
set -euo pipefail
MODE=${1:-uv}
VLLM=0.30.0
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== python3-dev: без Python.h torch.compile внутри vLLM падает на первом запуске"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3-dev >/dev/null 2>&1 && echo "Python.h: ок"

echo "== uv"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; }
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== веса Qwen2.5 (ревизии те же, что в лекциях)"
uv tool install -q "huggingface_hub[cli]" 2>/dev/null || true
hf download Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 >/dev/null
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 >/dev/null
du -sh ~/.cache/huggingface/hub/models--Qwen--* | sed 's#.*/##'

case $MODE in
  docker)
    echo "== образ vllm/vllm-openai:v$VLLM"
    docker pull -q vllm/vllm-openai:v$VLLM
    docker run --rm --gpus all vllm/vllm-openai:v$VLLM --version 2>/dev/null | tail -1 || true
    ;;
  uv)
    echo "== vLLM $VLLM в /opt/vllm"
    [ -d /opt/vllm ] || uv venv -q --python 3.12 /opt/vllm
    uv pip install -q --python /opt/vllm/bin/python "vllm==$VLLM"
    /opt/vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    ;;
esac

echo "== адаптеры"
ls -la ~/adapters/*/adapter_model.safetensors 2>/dev/null || echo "ещё нет: scp -r gpu_week5/returned/results/adapters root@<host>:~/adapters"
echo "готово"
