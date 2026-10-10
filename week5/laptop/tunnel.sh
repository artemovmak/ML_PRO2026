#!/usr/bin/env bash
# Шелл на машине плюс туннель: порт 8000 машины становится портом 8000 ноутбука. Пока окно открыто,
# с ноутбука работает http://localhost:8000. Второй туннель на тот же порт не поднимется: одно окно.
#
#   bash laptop/tunnel.sh [локальный порт, по умолчанию 8000]
set -euo pipefail
cd "$(dirname "$0")/.."
source laptop/machine.env
exec ssh -i "$K" -o IdentitiesOnly=yes -o ServerAliveInterval=30 -p "$P" -L "${1:-8000}:127.0.0.1:8000" "$H"
