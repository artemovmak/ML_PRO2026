#!/usr/bin/env bash
# Платформа одной командой: kind с портом 80, Traefik, MLflow, Airflow, S3 (RustFS), Prometheus и Grafana.
# Повторный запуск безопасен: что уже стоит, только обновится.
#
#   bash platform/up.sh                    кластер sem4
#   CLUSTER=sem3 bash platform/up.sh       другое имя кластера (то же нужно в ci.yml, KIND_CLUSTER)
#
# Ключи S3 по умолчанию учебные. Свои: S3_KEY=... S3_SECRET=... bash platform/up.sh
set -euo pipefail
CLUSTER=${CLUSTER:-sem4}
S3_KEY=${S3_KEY:-dvcadmin}
S3_SECRET=${S3_SECRET:-dvc-secret-123}
cd "$(dirname "$0")/.."

echo "== кластер $CLUSTER"
kind get clusters | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER" --config platform/kind-config.yaml
kubectl config use-context "kind-$CLUSTER" >/dev/null

echo "== образы, скачанные заранее (docker pull), кладём прямо в узел"
# kind load docker-image падает на мультиплатформенных образах, если Docker Desktop хранит их в containerd
# ("content digest ... not found"), поэтому через архив одной платформы
ARCH=$(docker version --format '{{.Server.Arch}}')
TMP=$(mktemp -d)
for img in apache/airflow:3.3.2 ghcr.io/mlflow/mlflow:v3.16.1 rustfs/rustfs:1.0.0; do
  if docker image inspect "$img" >/dev/null 2>&1; then
    docker save --platform "linux/$ARCH" "$img" -o "$TMP/image.tar" \
      && kind load image-archive "$TMP/image.tar" --name "$CLUSTER" >/dev/null && echo "  $img"
  fi
done
rm -rf "$TMP"

echo "== MLflow, S3, Airflow"
kubectl apply -f platform/mlflow.yaml
kubectl -n mlops create secret generic s3-credentials \
  --from-literal=AWS_ACCESS_KEY_ID="$S3_KEY" --from-literal=AWS_SECRET_ACCESS_KEY="$S3_SECRET" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f platform/s3.yaml -f platform/airflow.yaml

echo "== Traefik, Prometheus и Grafana"
helm repo add traefik https://traefik.github.io/charts >/dev/null 2>&1 || true
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm upgrade --install traefik traefik/traefik --version 41.6.0 \
  -n traefik --create-namespace -f platform/traefik-values.yaml --wait
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack --version 91.5.2 \
  -n monitoring --create-namespace -f platform/monitoring-values.yaml
kubectl apply -f platform/ingress.yaml

echo "== ждём поды"
kubectl -n mlops rollout status deploy/mlflow deploy/rustfs deploy/airflow --timeout=15m
kubectl -n monitoring rollout status deploy/monitoring-grafana --timeout=15m

echo "== бакет dvc в S3"
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT --aws-sigv4 "aws:amz:us-east-1:s3" \
    --user "$S3_KEY:$S3_SECRET" http://s3.localhost/dvc) || code=000
  case $code in 200|409) echo "бакет dvc есть ($code)"; break ;; esac
  sleep 2
done

echo
echo "Готово:  http://mlflow.localhost  http://airflow.localhost  http://grafana.localhost (admin/admin)"
echo "         http://s3-console.localhost/rustfs/console/index.html ($S3_KEY / $S3_SECRET)"
