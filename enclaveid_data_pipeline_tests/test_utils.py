import polars as pl


def make_simulated_dataset(
    counts_df: pl.DataFrame, activity_df: pl.DataFrame
) -> pl.DataFrame:
    """Given a data frame with the counts of activity per day (counts_df) and a
    data frame with raw activity (activity_df), this samples from activity_df
    to generate a simulated dataset with daily acitivty counts matching those
    in counts_df"""
    dfs = []
    for row in counts_df.iter_rows(named=True):
        dfs.append(
            activity_df.sample(row["count"]).with_columns(
                time=pl.lit(row["day"]).dt.combine(pl.col("time").dt.time())
            )
        )

    output_df = pl.concat(dfs)

    invalid_rows = (
        output_df.group_by(pl.col("time").dt.date())
        .count()
        .join(counts_df, left_on="time", right_on="day")
        .filter(pl.col("count") != pl.col("count_right"))
    )

    if invalid_rows.height > 0:
        raise ValueError(
            "The simulated data set does not match the daily counts provided."
        )

    return output_df
