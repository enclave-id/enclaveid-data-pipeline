from dagster import EnvVar
from dagster_polars import PolarsParquetIOManager

from ..consts import DAGSTER_STORAGE_BUCKET
from .mistral_resource import MistralResource
from .postgres_resource import PostgresClientResource

parquet_io_manager = PolarsParquetIOManager(
    extension=".snappy", base_dir=str(DAGSTER_STORAGE_BUCKET)
)

mistral_resource = MistralResource(api_key=EnvVar("MISTRAL_API_KEY"))

postgres_resource = PostgresClientResource(
    host=EnvVar("PGHOST"),
    port=EnvVar.int("PGPORT"),
    user=EnvVar("PGUSER"),
    password=EnvVar("PGPASSWORD"),
    database=EnvVar("PGDATABASE"),
    drivername="postgresql",
)
