from functools import partial

import cuml
import cupy as cp
import numpy as np
import polars as pl
from cuml.metrics import pairwise_distances
from dagster import AssetExecutionContext, asset
from hdbscan import HDBSCAN
from pydantic import Field
from sentence_transformers import SentenceTransformer

from ..partitions import user_partitions_def
from ..utils.custom_config import RowLimitConfig
from ..utils.old_history_utils import get_embeddings, get_full_history_sessions

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
        first_instruction=(
            "Here is a list of my Google search data. Are there any highly sensitive "
            "psychosocial interests?"
        ),
        second_instruction=(
            "Summarize the previous answer as a comma-separated array of strings. "
            "Only include highly sensitive psychosocial data."
        ),
        logger=context.log,
    )

    context.add_output_metadata(
        {"count_invalid_responses": sessions_output.count_invalid_responses}
    )

    return sessions_output.output_df


@asset(partitions_def=user_partitions_def, io_manager_key="parquet_io_manager")
def sensitive_interest_embeddings(
    context: AssetExecutionContext,
    config: RowLimitConfig,
    sensitive_interests: pl.DataFrame,
) -> pl.DataFrame:
    df = (
        # Enforce row_limit (if any)
        sensitive_interests.slice(0, config.row_limit)
        .select("date", "interests")
        # Explode the interests so we get the embeddings for each individual interest
        .explode("interests")
    )

    context.log.info("Loading the model. This may take a few minutes...")
    model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")

    context.log.info("Computing embeddings")
    return df.with_columns(
        embeddings=pl.col("interests").map_batches(
            partial(get_embeddings, model=model),
        )
    )


@asset(partitions_def=user_partitions_def, io_manager_key="parquet_io_manager")
def sensitive_interest_clusters(
    context: AssetExecutionContext,
    config: RowLimitConfig,
    sensitive_interest_embeddings: pl.DataFrame,
) -> pl.DataFrame:
    # Apply the row limit (if any)
    df = sensitive_interest_embeddings.slice(0, config.row_limit)

    # Convert the embeddings to a CuPy array
    embeddings_gpu = cp.asarray(df["embeddings"].to_numpy())

    # Reduce the embeddings dimensions
    umap_model = cuml.UMAP(
        n_neighbors=15, n_components=100, min_dist=0.1, metric="cosine"
    )
    reduced_data_gpu = umap_model.fit_transform(embeddings_gpu)

    # TODO: Implement a search across cluster_selection_epsilon to ensure a max
    # of 50 clusters are returned.

    # Compute the pairwise distances between the interests
    cosine_dist = pairwise_distances(reduced_data_gpu, metric="cosine")

    # Make the clusters
    clusterer = HDBSCAN(
        min_cluster_size=5,
        gen_min_span_tree=True,
        metric="precomputed",
        cluster_selection_epsilon=0.02,
    )
    cluster_labels = clusterer.fit_predict(cosine_dist.astype(np.float64).get())

    context.add_output_metadata(
        {
            "num_clusters": len(np.unique(cluster_labels)),
            "cluster_names": np.unique(cluster_labels).tolist(),
        }
    )

    # TODO: Implement logic to extract the top 5 largest clusters, and top 5 clusters
    # by farthest point sampling (most distance from each other -.e., cross-cluster distance)
    return df.with_columns(cluster_label=pl.Series(cluster_labels))
