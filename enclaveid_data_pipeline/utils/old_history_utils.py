import datetime
import gc
import re
from dataclasses import dataclass
from logging import Logger
from typing import Any

import polars as pl
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

SUMMARY_PROMPT = (
    "Here is a list of my Google search data. Are there any highly sensitive "
    "psychosocial interests? Summarize the answer as a comma-separated array of "
    "strings. Only include highly sensitive psychosocial data."
)


@dataclass
class FullHistorySessionsOutput:
    output_df: pl.DataFrame
    count_invalid_responses: int


@dataclass
class DailyInterestsGenerator:
    date: datetime.date
    df: pl.DataFrame
    llm: LLM
    sampling_params: SamplingParams
    chunk_size: int = 15

    def _extract_interests_list(self, text):
        match = re.search(r"\[(.*?)\]", text)
        if match:
            # If a match is found, split the substring by comma
            interests = match.group(1).replace('"', "").replace("'", "").split(",")
            return [s.strip() for s in interests]
        else:
            return None

    def _generate_chunked_prompts(self):
        # Keep only the relevant columns
        filtered_df = self.df.select("hour", "title")

        prompts: list[str] = []
        for idx, frame in enumerate(
            filtered_df.iter_slices(n_rows=self.chunk_size), start=1
        ):
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
                text = f"<s>[INST] {SUMMARY_PROMPT}\n\n{frame}[/INST]"
                prompts.append(text)

        self.chunked_prompts = prompts

    def _generate_chunked_responses(self):
        # TODO: We could potentially see a signifcant speedup by processing all
        # prompts at the same time, instead of daily, where batch sizes can be
        # rather small. This will raise some complexity with respect to mapping
        # responses to their date.
        output_requests = self.llm.generate(self.chunked_prompts, self.sampling_params)
        self.chunked_responses = [resp.outputs[0].text for resp in output_requests]

    def _generate_chunked_interests(self):
        self.chunked_interests = [
            self._extract_interests_list(resp) for resp in self.chunked_responses
        ]

    def generate_output_record(self):
        self._generate_chunked_prompts()
        self._generate_chunked_responses()
        self._generate_chunked_interests()

        # Filter out invalid responses
        valid_chunks = [chunk for chunk in self.chunked_interests if chunk is not None]

        # Flatten the chunks
        merged_interests = [interest for chunk in valid_chunks for interest in chunk]

        return {
            "date": self.date,
            "chunked_prompts": self.chunked_prompts,
            "chunked_responses": self.chunked_responses,
            "chunked_interests": self.chunked_interests,
            # Different chunks may have the same interests; we only want the
            # distinct interests across all chunks
            "interests": list(set(merged_interests)),
            "count_invalid_responses": (
                len(self.chunked_responses) - len(valid_chunks)
            ),
        }


def get_full_history_sessions(
    daily_dfs: dict[datetime.date, pl.DataFrame],
    chunk_size: int,
    logger: Logger,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
):
    llm = LLM(model=model_name)

    # TODO: We could potentially make this part of the Config so the params can be
    # configured from the Dagster UI
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

    daily_records: list[dict[str, Any]] = []
    for idx, (day, day_df) in enumerate(daily_dfs.items(), start=1):
        logger.info(f"Processing {day} ({idx}/{len(daily_dfs)})")
        interests_generator = DailyInterestsGenerator(
            date=day,
            df=day_df,
            llm=llm,
            sampling_params=sampling_params,
            chunk_size=chunk_size,
        )

        daily_records.append(interests_generator.generate_output_record())

    logger.info("Unloading the LLM and freeing GPU memory.")
    destroy_model_parallel()
    # del llm.llm_engine.driver_worker  # type: ignore
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()

    return FullHistorySessionsOutput(
        output_df=pl.from_records(daily_records),
        # Sum the invalid responses across all days
        count_invalid_responses=sum(
            out["count_invalid_responses"] for out in daily_records
        ),
    )
