from pathlib import Path

import pandas as pd
from dagster import AssetExecutionContext, AutoMaterializePolicy, asset

from enclaveid_data_pipeline.consts import PRODUCTION_STORAGE_BUCKET

from ..partitions import user_partitions_def


@asset(
    partitions_def=user_partitions_def,
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def parsed_takeout(context: AssetExecutionContext):
    f = PRODUCTION_STORAGE_BUCKET / context.partition_key
    context.log.info(f"{f} exists: {f.exists()}")

    # TODO: Temporarily return without processing
    return
