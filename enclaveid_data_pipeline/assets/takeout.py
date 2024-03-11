from textwrap import dedent

import pandas as pd
import polars as pl
from dagster import (
    AssetExecutionContext,
    AssetOut,
    Config,
    multi_asset,
)
from pydantic import Field

from enclaveid_data_pipeline.consts import PRODUCTION_STORAGE_BUCKET

from ..partitions import user_partitions_def


class TakeoutConfig(Config):
    threshold: str = Field(
        # TODO: Change back to -3mo before deployment.
        default="-15d",
        description=dedent(
            """
            The threshold for determining if data is 'old' or 'recent'. See 
            the docs for `polars.Expr.dt.offset_by` for examples of Polars' 
            "time offset language. 
            
            Note that it should always begin with a negative sign since we want
            to offset a negative amount from the date of the last record.

            '-3mo' by default, meaning any records within 3 months of the last
            record is considered "recent".
            """
        ),
    )


@multi_asset(
    outs={
        "full_takeout": AssetOut(
            key_prefix=["parsed"], io_manager_key="parquet_io_manager"
        ),
        "recent_takeout": AssetOut(
            key_prefix=["parsed"], io_manager_key="parquet_io_manager"
        ),
    },
    partitions_def=user_partitions_def,
)
def parsed_takeout(
    context: AssetExecutionContext, config: TakeoutConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parses the raw Takeout data an splits it into two sets based on recency.
    The exact threshold is controlled by the TakeoutConfig.threshold parameter
    (-3mo by default).
    """
    f = PRODUCTION_STORAGE_BUCKET / context.partition_key / "MyActivity.json"

    # TODO: Temporarily using Pandas to read the JSON because Polars doesn't
    # play well with UPath. Will fix this later.
    df = pl.from_pandas(pd.read_json(f), schema_overrides={"time": pl.Datetime}).select(
        pl.all().exclude("time"),
        timestamp=pl.col("time"),
        date=pl.col("time").dt.date(),
        hour=pl.col("time").dt.strftime("%H:%M"),
        month=pl.col("time").dt.strftime("%Y-%m-%d"),
    )

    return df, df.filter(
        pl.col("timestamp") > pl.col("timestamp").max().dt.offset_by(config.threshold)
    )
