import polars as pl
from dagster import AssetExecutionContext, asset
from pydantic import Field

from ..partitions import user_partitions_def
from ..utils.custom_config import RowLimitConfig
from ..utils.old_history_utils import get_full_history_sessions

SUMMARY_PROMPT = (
    "Here is a list of my Google search data. Are there any highly sensitive "
    "psychosocial interests? Summarize the answer as a comma-separated array of "
    "strings. Only include highly sensitive psychosocial data."
)


class AllSessionsConfig(RowLimitConfig):
    model_name: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description=(
            "The Hugging Face model to use as the LLM. See the vLLMs docs for a "
            "list of the support models:\n"
            "https://docs.vllm.ai/en/latest/models/supported_models.html"
        ),
    )

    chunk_size: int = Field(
        default=15,
        description=(
            "Split the raw history into chunks of this size. We allow vLLM to "
            "determine the ideal batch size by itsef, so this has no impact on "
            "runtime but it still determines how many records are shown to the "
            "LLM at one time. Having too many records can cause the LLM to give "
            "sub-par responses."
        ),
    )


@asset(partitions_def=user_partitions_def, io_manager_key="parquet_io_manager")
def sensitive_interests(
    context: AssetExecutionContext,
    config: AllSessionsConfig,
    full_takeout: pl.DataFrame,
) -> pl.DataFrame:
    # Enforce the row_limit (if any) per day and sort the data by time because
    # Polars reads data out-of-order
    full_takeout = full_takeout.slice(0, config.row_limit).sort("timestamp")

    # Split into multiple data frames (one per day). This is necessary to correctly
    # identify the data associated with each time entry.
    daily_dfs = full_takeout.with_columns(
        date=pl.col("timestamp").dt.date()
    ).partition_by("date", as_dict=True, include_key=False)

    sessions_output = get_full_history_sessions(
        daily_dfs=daily_dfs,
        chunk_size=config.chunk_size,
        prompt_prefix=(
            "Here is a list of my Google search data. Are there any highly sensitive "
            "psychosocial interests?"
        ),
        prompt_suffix=(
            "Summarize the answer as a comma-separated array of strings. Only "
            "include highly sensitive psychosocial data."
        ),
        logger=context.log,
    )

    context.add_output_metadata(
        {"count_invalid_responses": sessions_output.count_invalid_responses}
    )

    return sessions_output.output_df
