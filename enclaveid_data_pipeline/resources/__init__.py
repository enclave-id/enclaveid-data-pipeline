from dagster_polars import PolarsParquetIOManager

from ..consts import DAGSTER_STORAGE_BUCKET

parquet_io_manager = PolarsParquetIOManager(
    extension=".snappy", base_dir=str(DAGSTER_STORAGE_BUCKET)
)
