import datetime
import json
import math
import os
from dataclasses import dataclass
from functools import partial
from textwrap import dedent

import polars as pl
import psycopg
from dagster import AssetExecutionContext, DagsterLogManager, asset
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pgvector.psycopg import register_vector
from pydantic import Field

from ..partitions import user_partitions_def
from ..resources.mistral_resource import MistralResource
from ..resources.postgres_resource import PGVectorClient, PGVectorClientResource
from ..utils.custom_config import RowLimitConfig

# TODO: Avoid auto-committing queries and bundle them as one transaction,
# where appropriate.

SUMMARY_PROMPT = dedent("""
    Analyze the provided list of Google search records to identify distinct topic groups. For each group, create a summary in the JSON format below. Ensure each summary includes: 

    - `time_start`: The start time of the first search in the group.
    - `time_end`: The end time of the last search in the group.
    - `description`: A detailed account of the searches and site visits, enriched with inferred user intent and additional insights into the topic.
    - `interests`: A list of keywords representing the user's interests based on the searches.

    Each `description` should not only recap the searches but also offer a deeper understanding of what the user might be seeking or the broader context of their inquiries. Group searches based on thematic relevance and timing. 

    Example of JSON output format:

    {
    "time_start": "HH:MM",
    "time_end": "HH:MM",
    "description": "Elaborate on what the user did and why, based on the search terms and visited pages.",
    "interests": ["keyword1", "keyword2"]
    }
    
    Here is a list of searches:
""")


# function to turn string returned from the LLM into valid python dictionary
def extract_json(text):
    # Helper function to find the matching closing brace or bracket
    def find_closing(text, open_pos, open_char, close_char):
        balance = 0
        for i in range(open_pos, len(text)):
            if text[i] == open_char:
                balance += 1
            elif text[i] == close_char:
                balance -= 1
                if balance == 0:
                    return i
        return -1

    # Find the start of the JSON object/array
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return {}, None  # No JSON found

    # Determine which comes first or use -1 if not found
    start_index = (
        obj_start
        if arr_start == -1 or (obj_start != -1 and obj_start < arr_start)
        else arr_start
    )
    open_char = "{" if start_index == obj_start else "["
    close_char = "}" if open_char == "{" else "]"

    # Find the matching closing brace/bracket
    end_index = find_closing(text, start_index, open_char, close_char)

    if start_index != -1 and end_index != -1:
        json_text = text[start_index : end_index + 1]
        try:
            json_response = json.loads(json_text)
            return json_response, text[end_index + 1 :]
        except json.JSONDecodeError:
            return {}, None  # Handle invalid JSON
    else:
        return {}, None


# TODO: Consider making this a method of the MistralResource.
def get_completion(prompt, client: MistralClient, model="mistral-tiny"):
    messages = [ChatMessage(role="user", content=prompt)]

    chat_response = client.chat(
        model=model,
        messages=messages,
    )

    return chat_response.choices[0].message.content


@dataclass
class ChunkedSessionOutput:
    output_df: pl.DataFrame
    raw_answers: list[str | list[str]]
    num_sessions: int
    invalid_types: int
    invalid_keys: int
    invalid_times: int
    invalid_sessions: int


def get_daily_sessions(
    df: pl.DataFrame,
    client: MistralClient,
    chunk_size: int,
    logger: DagsterLogManager,
    day: datetime.date,
) -> ChunkedSessionOutput:
    df = df.select("hour", "title")
    num_chunks = math.ceil(df.height / chunk_size)

    raw_answers = []
    sessions_list = []

    # TODO: Make this async / concurrent.
    for idx, frame in enumerate(df.iter_slices(n_rows=chunk_size), start=1):
        logger.info(f"[{day}]  Processing chunk {idx} / {num_chunks}")

        # Set Polars' string formatting so none of the rows or strings are
        # compressed / cut off.
        max_chars = frame["title"].str.len_chars().max()
        with pl.Config(
            tbl_formatting="NOTHING",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            fmt_str_lengths=max_chars,
            tbl_rows=-1,
        ):
            answer = get_completion(f"{SUMMARY_PROMPT}\n{frame}", client)
            raw_answers.append(answer)

        # Sometimes the LLM returns multipe json objects in a list
        # Some other times it returns a single json object
        # We need to handle both cases
        while answer:
            parsed_result, answer = extract_json(answer)

            if parsed_result:
                if isinstance(parsed_result, dict):
                    sessions_list.append(parsed_result)
                elif isinstance(parsed_result, list):
                    sessions_list.extend(parsed_result)

    all_sessions = len(sessions_list)

    # Filter out responses with the wrong type
    sessions_list = [x for x in sessions_list if isinstance(x, dict)]
    valid_types = len(sessions_list)
    invalid_types = all_sessions - valid_types

    # Filter out responses with the wrong JSON format/keys
    sessions_list = [
        d
        for d in sessions_list
        if d.keys() == {"time_start", "time_end", "description", "interests"}
    ]
    valid_keys = len(sessions_list)
    invalid_keys = valid_types - valid_keys

    output = (
        pl.from_dicts(
            sessions_list,
            schema={
                "time_start": pl.Utf8,
                "time_end": pl.Utf8,
                "description": pl.Utf8,
                "interests": pl.List(pl.Utf8),
            },
        )
        # Filter out any rows with invalid time strings.
        .filter(
            pl.col("time_end").str.contains(r"^\d{2}:\d{2}$")
            & pl.col("time_start").str.contains(r"^\d{2}:\d{2}$")
        )
        # Cast the times from string to pl.Time and add the date
        .with_columns(
            pl.col("time_end").str.strptime(pl.Time, "%H:%M"),
            pl.col("time_start").str.strptime(pl.Time, "%H:%M"),
            date=pl.lit(day),
        )
    )

    invalid_times = valid_keys - len(output)
    return ChunkedSessionOutput(
        output_df=output,
        raw_answers=raw_answers,
        num_sessions=all_sessions,
        invalid_types=invalid_types,
        invalid_keys=invalid_keys,
        invalid_times=invalid_times,
        invalid_sessions=invalid_types + invalid_keys + invalid_times,
    )


class SessionsConfig(RowLimitConfig):
    chunk_size: int = Field(default=15, description="The size of each chunk.")


# TODO: Consider converting all these assets into a single graph-backed asset
# called recent_sessions.
@asset(partitions_def=user_partitions_def, io_manager_key="parquet_io_manager")
def recent_sessions(
    context: AssetExecutionContext,
    config: SessionsConfig,
    mistral: MistralResource,
    recent_takeout: pl.DataFrame,
) -> pl.DataFrame:
    # Enforce the row_limit (if any) per day
    recent_takeout = recent_takeout.slice(0, config.row_limit)

    client = mistral.get_client()

    # Sort the data by time -- Polars might read data out-of-order
    recent_sessions = recent_takeout.sort("timestamp")

    # Split into multiple data frames (one per day). This is necessary to correctly
    # identify the data associated with each time entry.
    daily_dfs = recent_sessions.with_columns(
        date=pl.col("timestamp").dt.date()
    ).partition_by("date", as_dict=True, include_key=False)

    # TODO: Make this (and the calls inside get_daily_sessions) async/concurrent.
    daily_outputs: list[ChunkedSessionOutput] = []
    for day, day_df in daily_dfs.items():
        daily_outputs.append(
            get_daily_sessions(day_df, client, config.chunk_size, context.log, day)
        )

    context.add_output_metadata(
        {
            "num_sessions": sum(out.num_sessions for out in daily_outputs),
            "invalid_types": sum(out.invalid_types for out in daily_outputs),
            "invalid_keys": sum(out.invalid_keys for out in daily_outputs),
            "invalid_times": sum(out.invalid_times for out in daily_outputs),
            "invalid_sessions": sum(out.invalid_sessions for out in daily_outputs),
            "error_rate": round(
                sum(out.invalid_sessions for out in daily_outputs)
                / sum(out.num_sessions for out in daily_outputs),
                2,
            ),
        }
    )

    return pl.concat((out.output_df for out in daily_outputs))


# TODO: Consider making this a method of the MistralResource.
def get_embeddings(
    texts: pl.Series, client: MistralClient, chunk_size: int, logger: DagsterLogManager
) -> pl.Series:
    # TODO: Maybe we want to make this async?
    num_chunks = math.ceil(len(texts) / chunk_size)
    embeddings = []
    for idx in range(0, len(texts), chunk_size):
        logger.info(f"Processing chunk {int(idx/chunk_size) + 1} / {num_chunks}")

        response = client.embeddings(
            model="mistral-embed",
            input=texts.slice(idx, chunk_size).to_list(),
        )

        embeddings.extend(x.embedding for x in response.data)

    return pl.Series(embeddings, dtype=pl.Array(pl.Float64, 1024))


# TODO: Consider encapsulating this logic in an IOManager and/or moving the binary
# copy logic into the PGVectorClient.
def upload_embeddings(
    context: AssetExecutionContext, client: PGVectorClient, df: pl.DataFrame
):
    context.log.info(f"Flushing existing rows for partition: {context.partition_key}")
    with client._get_conn() as conn:
        with client._get_cursor(conn) as cur:
            cleanup_query = (
                f"DELETE FROM {context.asset_key.path[-1]} "
                f"WHERE user_id = '{context.partition_key}'"
            )
            context.log.debug(f"Executing query:\n{cleanup_query}")
            cur.execute(cleanup_query)  # type: ignore

    context.log.info(f"COPYing {len(df)} rows to Postgres...")
    col_list = (
        "user_id",
        "date",
        "time_start",
        "time_end",
        "description",
        "interests",
        "embedding",
    )

    df = df.select(col_list)

    copy_statement = (
        f"COPY {context.asset_key.path[-1]} "
        f"({', '.join(col_list)}) "
        "FROM STDIN "
        "WITH (FORMAT BINARY)"
    )

    # TODO: Optimization -- Using psycopg.AsyncConnection or asyncpg should speed this up
    num_rows = df.height
    with psycopg.connect(os.getenv("PSQL_URL", ""), autocommit=True) as conn:
        register_vector(conn)

        with conn.cursor().copy(copy_statement) as copy:  # type: ignore
            # Binary copy requires explicitly setting the types.
            # https://www.psycopg.org/psycopg3/docs/basic/copy.html#binary-copy
            copy.set_types(["text", "date", "time", "time", "text", "text[]", "vector"])

            for idx, r in enumerate(df.iter_rows(), start=1):
                copy.write_row(r)
                while conn.pgconn.flush() == 1:
                    pass

                if idx % 10 == 0 or idx == num_rows:
                    context.log.info(f"Finished copying {idx} / {num_rows} rows.")

    context.log.info("Finished COPY operation.")


class SessionEmbeddingsConfig(RowLimitConfig):
    chunk_size: int = Field(default=100, description="The size of each chunk.")


# TODO: Consider incorporate this logic inside recent_sessions.
@asset(
    partitions_def=user_partitions_def,
    io_manager_key="parquet_io_manager",
)
def recent_session_embeddings(
    context: AssetExecutionContext,
    config: SessionEmbeddingsConfig,
    mistral: MistralResource,
    pgvector: PGVectorClientResource,
    recent_sessions: pl.DataFrame,
) -> pl.DataFrame:
    # Enforce row_limit (if any)
    recent_sessions = recent_sessions.slice(0, config.row_limit)

    context.log.info("Getting embeddings...")
    client = mistral.get_client()
    recent_sessions = recent_sessions.with_columns(
        embedding=pl.col("description").map_batches(
            partial(
                get_embeddings,
                client=client,
                chunk_size=config.chunk_size,
                logger=context.log,
            )
        ),
        user_id=pl.lit(context.partition_key),
    )

    upload_embeddings(context, pgvector.get_client(), recent_sessions)
    return recent_sessions


@asset(
    partitions_def=user_partitions_def,
    deps=[recent_session_embeddings],
)
def time_threshold(
    context: AssetExecutionContext,
    pgvector: PGVectorClientResource,
):
    client = pgvector.get_client()
    # Calculate the time and similarity thresholds for this user
    results = client.execute_query(
        query=f"""
        WITH LaggedSessions AS (
            SELECT
                date,
                time_start,
                time_end,
                LAG(time_end) OVER (ORDER BY date, time_start) AS prev_time_end
            FROM
                recent_session_embeddings
            WHERE
                user_id = '{context.partition_key}'
        ),
        
        TimeDifferences AS (
            SELECT
                EXTRACT('epoch' FROM time_start - prev_time_end) AS time_diff
            FROM
                LaggedSessions
            WHERE
                time_start > prev_time_end
        )
        
        SELECT
            percentile_cont(0.10) WITHIN GROUP (ORDER BY time_diff) AS time_interval_10th
        FROM
            TimeDifferences
        """,
        fetch_results=True,
    )

    if results is None or len(results) == 0:
        raise Exception(
            f"Could not determine the time_threshold for {context.partition_key}."
        )

    time_threshold = results[0][0]
    context.log.info(f"{time_threshold = }")
    return time_threshold


@asset(
    partitions_def=user_partitions_def,
    deps=[recent_session_embeddings],
)
def similarity_threshold(
    context: AssetExecutionContext,
    pgvector: PGVectorClientResource,
):
    client = pgvector.get_client()
    # Calculate the time and similarity thresholds for this user
    results = client.execute_query(
        query=f"""
        WITH CosineSimilarities AS (
            SELECT
                date,
                time_start,
                1 - (embedding <=> LAG(embedding) OVER (ORDER BY date, time_start)) AS cosine_similarity
            FROM
                recent_session_embeddings
            WHERE
                user_id = '{context.partition_key}'
        ),

        FilteredSimilarities AS (
            SELECT
                cosine_similarity
            FROM
                CosineSimilarities
            WHERE
                cosine_similarity IS NOT NULL
        )
        
        SELECT
            percentile_cont(0.90) WITHIN GROUP (ORDER BY cosine_similarity) AS embedding_similarity_90th
        FROM
            FilteredSimilarities""",
        fetch_results=True,
    )

    if results is None or len(results) == 0:
        raise Exception(
            f"Could not determine the similarity_threshold for {context.partition_key}."
        )

    similarity_threshold = results[0][0]
    context.log.info(f"{similarity_threshold = }")
    return similarity_threshold


@asset(
    partitions_def=user_partitions_def,
    deps=[recent_session_embeddings],
)
def recent_sessions_merged(
    context: AssetExecutionContext,
    pgvector: PGVectorClientResource,
    time_threshold,
    similarity_threshold,
):
    client = pgvector.get_client()

    context.log.info(
        f"Time threshold: {time_threshold:.2f} seconds. "
        f"Embedding similarity threshold: {similarity_threshold:.4f}"
    )

    # Create a copy of the sessions in the merged table (but first delete any
    # existing rows for this user)
    client.execute_query(
        f"DELETE FROM recent_sessions_merged where user_id = '{context.partition_key}'"
    )
    client.execute_query(
        query=f"""
        INSERT INTO recent_sessions_merged 
        SELECT *
        FROM recent_session_embeddings 
        WHERE user_id = '{context.partition_key}';
        """
    )

    # TODO: Explore implementing this logic in Polars instead. The problems with
    # the current logic are noted in Issue #4.
    #
    # --------------------------------------------------------------------------
    #
    # TODO: Explore recomputing a new description and embedding for the merged
    # session (or concatenating the description strings and finding the cluster
    # mean of the embeddings of all the sessions that will be merged) as an
    # alternative solution.
    candidates_to_merge = client.execute_query(
        query=f"""
        SELECT 
            a.id, 
            b.id
        FROM 
            recent_sessions_merged a
        JOIN 
            recent_sessions_merged b 
            ON
                a.user_id = '{context.partition_key}'
                AND a.user_id = b.user_id
                AND a.id != b.id 
                AND (
                    b.date > a.date 
                    OR (a.date = b.date AND b.time_start >= a.time_end)
                )
        WHERE 
            EXTRACT(
                'epoch' FROM (
                    (b.date || ' ' || b.time_start)::timestamp
                    - (a.date || ' ' || a.time_end)::timestamp
                ) 
            ) <= {time_threshold}
            AND
            1 - (a.embedding <=> b.embedding) >= {similarity_threshold}""",
        fetch_results=True,
    )

    msg = "\n".join(str(pair) for pair in candidates_to_merge)  # type: ignore
    context.log.info(f"Merging sessions:\n{msg}")

    for a, b in candidates_to_merge:  # type: ignore
        # Update time_end of document a with the maximum time_end of both sessions
        client.execute_query(
            f"""
            UPDATE recent_sessions_merged
            SET time_end = (
                SELECT 
                    GREATEST(max_a.time_end, max_b.time_end)
                FROM 
                    (SELECT time_end FROM recent_sessions_merged WHERE id = {a}) as max_a,
                    (SELECT time_end FROM recent_sessions_merged WHERE id = {b}) as max_b
            )
            WHERE id = {a}
            """
        )

        # Update time_start of document a with the minimum time_start of both sessions
        client.execute_query(
            f"""
            UPDATE recent_sessions_merged
            SET time_start = (
                SELECT 
                    LEAST(min_a.time_start, min_b.time_start)
                FROM 
                    (SELECT time_start FROM recent_sessions_merged WHERE id = {a}) as min_a,
                    (SELECT time_start FROM recent_sessions_merged WHERE id = {b}) as min_b
            )
            WHERE id = {a}
            """
        )
        # Delete the duplicate session
        client.execute_query(f"DELETE FROM recent_sessions_merged WHERE id = {b}")


@asset(
    partitions_def=user_partitions_def,
    deps=[recent_sessions_merged],
)
def recent_sessions_graph(
    context: AssetExecutionContext,
    pgvector: PGVectorClientResource,
    similarity_threshold,
):
    client = pgvector.get_client()

    # Cleanup existing data for this partition (i.e., user)
    client.execute_query(
        f"DELETE FROM recent_sessions_graph where user_id = '{context.partition_key}'"
    )

    # Update the graph with the edges for this partition (i.e., user)
    client.execute_query(
        f"""
        WITH DocumentPairs AS (
            SELECT
                a.user_id,
                a.id AS doc_id,
                b.id AS compared_doc_id,
                1 - (a.embedding <=> b.embedding) AS similarity,
                a.date AS doc_date,
                b.date AS compared_doc_date,
                a.time_end AS doc_time_end,
                b.time_start AS compared_doc_time_start
            FROM
                recent_sessions_merged a
            JOIN
                recent_sessions_merged b 
                ON
                    a.user_id = '{context.partition_key}'
                    AND a.user_id = b.user_id
                    AND a.id != b.id 
                    AND (
                        b.date > a.date 
                        OR (a.date = b.date AND b.time_start >= a.time_end)
                    )
        ),

        RankedPairs AS (
            SELECT
                *,
                ROW_NUMBER() OVER(
                    PARTITION BY user_id, doc_id
                    ORDER BY similarity DESC
                ) AS rank
            FROM
                DocumentPairs
        ), 

        FilteredPairs1 as (
            SELECT
                user_id,
                doc_id,
                compared_doc_id,
                similarity
            FROM
                RankedPairs
            WHERE
                rank = 1
                AND similarity > {similarity_threshold}
        )

        INSERT INTO recent_sessions_graph (user_id, parent_id, child_id, weight)
        
        SELECT
            user_id,
            doc_id,
            compared_doc_id,
            1 - similarity as distance
        FROM
            FilteredPairs1
        """
    )
