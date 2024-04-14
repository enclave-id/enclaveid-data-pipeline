from textwrap import dedent

import polars as pl
from dagster import (
    AssetExecutionContext,
    AssetOut,
    multi_asset,
)
from pydantic import Field

from ..consts import PRODUCTION_STORAGE_BUCKET
from ..partitions import user_partitions_def
from ..utils.custom_config import RowLimitConfig


class TakeoutConfig(RowLimitConfig):
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
    """Parses the raw Takeout data and splits it into two sets based on recency.
    The exact threshold is controlled by the TakeoutConfig.threshold parameter
    (-3mo by default).
    """
    if not config.threshold.startswith("-"):
        raise ValueError("the `threshold` should always start with a `-` sign.")

    p = PRODUCTION_STORAGE_BUCKET / context.partition_key / "MyActivity.json"

    with p.open("rb") as f:
        raw_df = pl.read_json(f.read(), schema_overrides={"time": pl.Datetime})

    full_df = raw_df.select(
        pl.all().exclude("time"),
        timestamp=pl.col("time"),
        date=pl.col("time").dt.date(),
        hour=pl.col("time").dt.strftime("%H:%M"),
        month=pl.col("time").dt.strftime("%Y-%m-%d"),
    )

    recent_df = full_df.filter(
        pl.col("timestamp") > pl.col("timestamp").max().dt.offset_by(config.threshold)
    )

    # The row_limit is enforced on the output instead of the input because prematurely
    # filtering the rows might leave no "recent" rows in the output, making recent_df empty.
    return full_df.slice(0, config.row_limit), recent_df.slice(0, config.row_limit)
