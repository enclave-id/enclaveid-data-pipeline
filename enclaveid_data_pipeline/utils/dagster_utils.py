from dagster import (
    AssetExecutionContext,
    AssetKey,
    DagsterEventType,
    EventRecordsFilter,
)
from dagster._core.events import StepMaterializationData


def get_latest_materialization_metadata(
    context: AssetExecutionContext, asset_key: AssetKey, partition_key: str
):
    latest_observation_record = context.instance.get_event_records(
        EventRecordsFilter(
            event_type=DagsterEventType.ASSET_OBSERVATION,
            asset_key=asset_key,
            asset_partitions=[partition_key],
        ),
        ascending=False,
        limit=1,
    )[0]
    event = latest_observation_record.event_log_entry.dagster_event

    if event is None:
        return

    event_data = event.event_specific_data

    if event_data is None or not isinstance(event_data, StepMaterializationData):
        return

    return event_data.materialization.metadata
