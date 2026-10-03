from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import CronTriggerTimetable, Param, dag, task
from kubernetes.client import models as k8s

PULL = "uv run --no-sync dvc pull -r cluster datasets/online_retail.parquet && "
MLFLOW = {"MLFLOW_TRACKING_URI": "http://mlflow.mlops:5000"}
COMMON = dict(namespace="mlops", in_cluster=True, image="{{ var.value.churn_image }}", image_pull_policy="IfNotPresent",
              get_logs=True, on_finish_action="delete_pod")

def secret(name: str) -> list:
    return [k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name=name))]

@dag(
    # каждый день в 04:00 по Москве, через час после churn_retrain; пропущенные запуски не догоняем
    schedule=CronTriggerTimetable("0 4 * * *", timezone="Europe/Moscow"),
    catchup=False,
    params={"factors": Param(64, type="integer", minimum=8, maximum=256, description="размер эмбеддинга ALS")},
    tags=["recs"],
)
def recs_retrain():
    validate = KubernetesPodOperator(
        task_id="validate", name="recs-validate", env_vars=MLFLOW, env_from=secret("s3-credentials"),
        cmds=["sh", "-c", PULL + "uv run --no-sync python -m churn.recs.data"], **COMMON,
    )
    train = KubernetesPodOperator(
        task_id="train", name="recs-train", env_vars={**MLFLOW, "FACTORS": "{{ params.factors }}"},
        env_from=secret("s3-credentials"), do_xcom_push=True,
        cmds=["sh", "-c", PULL + "uv run --no-sync python -m churn.recs.train"], **COMMON,
    )

    @task.short_circuit
    def promoted(result: dict) -> bool:
        print(f"версия {result['version']}, NDCG@10 {result['ndcg_10']}, проверки гейта {result['checks']}")
        return result["promoted"]

    publish = KubernetesPodOperator(
        task_id="publish", name="recs-publish", env_vars=MLFLOW, env_from=secret("recs-db"), do_xcom_push=True,
        cmds=["uv", "run", "--no-sync", "python", "-m", "churn.recs.publish"], **COMMON,
    )

    validate >> train
    promoted(train.output) >> publish


recs_retrain()
