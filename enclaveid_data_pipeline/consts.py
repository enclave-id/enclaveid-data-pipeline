import os

from upath import UPath


def get_environment() -> str:
    if os.getenv("DAGSTER_CLOUD_IS_BRANCH_DEPLOYMENT", "") == "1":
        return "BRANCH"
    if os.getenv("DAGSTER_CLOUD_DEPLOYMENT_NAME", "") == "prod":
        return "PROD"
    return "LOCAL"


DEPLOYMENT_TYPE = get_environment()

PRODUCTION_STORAGE_BUCKET: UPath = {
    "LOCAL": UPath(__file__).parent.parent / "data",
    "BRANCH": UPath("my-staging-bucket"),
    "PROD": UPath("my-production-bucket"),
}[DEPLOYMENT_TYPE]

DAGSTER_STORAGE_BUCKET = {
    "LOCAL": UPath("/tmp/dagster_data"),
    "BRANCH": UPath("my-dagster-staging-bucket"),
    "PROD": UPath("my-dagster-prod-bucket"),
}[DEPLOYMENT_TYPE]
