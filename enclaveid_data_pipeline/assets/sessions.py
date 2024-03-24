import json
import math
import os
from functools import partial
from textwrap import dedent

import polars as pl
import psycopg
from dagster import AssetExecutionContext, asset
from mistralai.models.chat_completion import ChatMessage
from pgvector.psycopg import register_vector
from pydantic import Field

from ..partitions import user_partitions_def
from ..resources.mistral_resource import MistralResource
from ..utils.custom_config import RowLimitConfig

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
def get_completion(prompt, client, model="mistral-tiny"):
    messages = [ChatMessage(role="user", content=prompt)]

    chat_response = client.chat(
        model=model,
        messages=messages,
    )

    return chat_response.choices[0].message.content


class SessionsConfig(RowLimitConfig):
    chunk_size: int = Field(default=15, description="The size of each chunk.")


@asset(partitions_def=user_partitions_def, io_manager_key="parquet_io_manager")
def recent_sessions(
    context: AssetExecutionContext,
    config: SessionsConfig,
    mistral: MistralResource,
    recent_takeout: pl.DataFrame,
) -> pl.DataFrame:
    # Sort the data by time -- Polars might read data out-of-order
    df = recent_takeout.sort("hour").select("title", "hour")

    # Enforce the row_limit (if any)
    df = df.slice(0, config.row_limit)
    num_chunks = math.ceil(recent_takeout.height / config.chunk_size)

    client = mistral.get_client()

    # TODO: Make this async / concurrent.
    sessions_list = []
    for idx, frame in enumerate(df.iter_slices(n_rows=config.chunk_size)):
        context.log.info(f"Processing chunk {idx + 1} / {num_chunks}")

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
    sessions_list = [
        d
        for d in sessions_list
        if d.keys() == {"time_start", "time_end", "description", "interests"}
    ]
    malformed_sessions = all_sessions - len(sessions_list)
    context.add_output_metadata(
        {
            "num_chunks": num_chunks,
            "malformed_chunks": malformed_sessions,
            "error_rate": round(malformed_sessions / all_sessions, 2),
        }
    )

    # Conver to a data frame and cast the times from string to pl.Time
    output = pl.from_dicts(sessions_list).with_columns(
        pl.col("time_end").str.strptime(pl.Time, "%H:%M"),
        pl.col("time_start").str.strptime(pl.Time, "%H:%M"),
    )

    return output


# TODO: Consider making this a method of the MistralResource.
def get_embeddings(texts: pl.Series, client) -> pl.Series:
    # TODO: Maybe we want to make this async?
    response = client.embeddings(
        model="mistral-embed",
        input=texts.to_list(),
    )

    return pl.Series(
        (x.embedding for x in response.data), dtype=pl.Array(pl.Float64, 1024)
    )


# TODO: Consider encapsulating this logic in an IOManager.
def upload_embeddings(df: pl.DataFrame, context: AssetExecutionContext):
    context.log.info(f"COPYing {len(df)} rows to Postgres...")

    copy_statement = (
        "COPY recent_sessions "
        "(user_id, description, time_start, time_end, interests, embedding) "
        "FROM STDIN "
        "WITH (FORMAT BINARY)"
    )

    # TODO: Optimization -- Using psycopg.AsyncConnection or asyncpg should speed this up
    num_rows = df.height
    with psycopg.connect(os.getenv("PSQL_URL", ""), autocommit=True) as conn:
        register_vector(conn)

        with conn.cursor().copy(copy_statement) as copy:
            # Binary copy requires explicitly setting the types.
            # https://www.psycopg.org/psycopg3/docs/basic/copy.html#binary-copy
            copy.set_types(["text", "text", "time", "time", "text[]", "vector"])

            for idx, r in enumerate(df.iter_rows(), start=1):
                copy.write_row(r)
                while conn.pgconn.flush() == 1:
                    pass

                if idx % 10 == 0 or idx == num_rows:
                    context.log.info(f"Finished copying {idx} / {num_rows} rows.")

    context.log.info("Finished COPY operation.")


@asset(partitions_def=user_partitions_def, io_manager_key="parquet_io_manager")
def recent_session_embeddings(
    context: AssetExecutionContext,
    config: RowLimitConfig,
    mistral: MistralResource,
    recent_sessions: pl.DataFrame,
) -> pl.DataFrame:
    # Enforce row_limit (if any)
    recent_sessions = recent_sessions.slice(0, config.row_limit)

    context.log.info("Getting embeddings...")
    client = mistral.get_client()
    recent_sessions = recent_sessions.with_columns(
        embeddings=pl.col("description").map_batches(
            partial(get_embeddings, client=client)
        ),
        user_id=pl.lit(context.partition_key),
    ).select(
        "user_id", "description", "time_start", "time_end", "interests", "embeddings"
    )

    upload_embeddings(recent_sessions, context)
    return recent_sessions


# FIXME: Next steps -- research pgvector and create an IOManager for storing the embeddings.
