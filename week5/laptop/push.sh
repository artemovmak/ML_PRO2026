#!/usr/bin/env bash
# С ноутбука: залить на свежую машину скрипты и адаптер, поставить vLLM и веса. Около 8 минут, повторный запуск безопасен.
#
#   bash laptop/push.sh            из папки week5, адрес и ключ в laptop/machine.env
set -euo pipefail
cd "$(dirname "$0")/.."
source laptop/machine.env
SSH="ssh -i $K -o IdentitiesOnly=yes -p $P $H"

echo "== доступ"
$SSH 'echo "ok: $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader)"'

echo "== файлы"
$SSH 'mkdir -p ~/seminar ~/adapters'
scp -q -i "$K" -o IdentitiesOnly=yes -P "$P" remote/*.sh "$H:~/seminar/"
scp -q -r -i "$K" -o IdentitiesOnly=yes -P "$P" adapters/reranker-1.5B "$H:~/adapters/"
$SSH 'sed -i "s/\r$//" ~/seminar/*.sh; ls ~/adapters/reranker-1.5B'

echo "== установка (лог на машине: ~/seminar/setup.log)"
T0=$(date +%s)
$SSH 'bash ~/seminar/setup.sh uv 2>&1 | tee ~/seminar/setup.log | grep --line-buffered -v "Hint\|Warning\|^$"'
echo "установка заняла $(( ($(date +%s) - T0) / 60 )) мин $(( ($(date +%s) - T0) % 60 )) с"
echo
echo "дальше: bash laptop/tunnel.sh (окно 1), в нём на машине: bash ~/seminar/serve.sh uv"
