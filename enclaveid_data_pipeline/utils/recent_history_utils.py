import datetime
import json
import math
from dataclasses import dataclass

import polars as pl
from dagster import DagsterLogManager
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


@dataclass
class ChunkedSessionOutput:
    output_df: pl.DataFrame
    raw_answers: list[str | list[str]]
    num_sessions: int
    invalid_types: int
    invalid_keys: int
    invalid_times: int
    invalid_sessions: int


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


def get_daily_sessions(
    df: pl.DataFrame,
    client: MistralClient,
    chunk_size: int,
    logger: DagsterLogManager,
    day: datetime.date,
    prompt: str,
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
            answer = get_completion(f"{prompt}\n{frame}", client)
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
