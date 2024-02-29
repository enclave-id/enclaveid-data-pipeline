import warnings

from dagster import Definitions, ExperimentalWarning, load_assets_from_modules

from . import assets

warnings.filterwarnings("ignore", category=ExperimentalWarning)

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
)
