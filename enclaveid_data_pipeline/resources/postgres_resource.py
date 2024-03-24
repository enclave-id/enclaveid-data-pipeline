from logging import Logger
from typing import Any

import dagster._check as check
import polars as pl
from dagster import get_dagster_logger
from dagster_aws.redshift import RedshiftClient, RedshiftClientResource
from sqlalchemy.engine import URL


# TODO: dagster_aws depends on psycopg2. Consider importing the original classes
# and upgrading to psycopg3.
class PostgresClient(RedshiftClient):
    def __init__(
        self,
        conn_args: dict[str, Any],
        autocommit: bool | None,
        log: Logger,
        uri: str,
    ):
        # Extract parameters from resource config
        self.conn_args = conn_args

        self.autocommit = autocommit
        self.log = log
        self.uri = uri

    def fetch(self, query: str) -> pl.DataFrame:
        check.str_param(query, "query")
        self.log.debug(f"Executing query:\n{query}")
        return pl.read_database_uri(query=query, uri=self.uri)


class PostgresClientResource(RedshiftClientResource):
    drivername: str

    def get_client(self) -> PostgresClient:
        conn_args = {
            k: getattr(self, k, None)
            for k in (
                "host",
                "port",
                "user",
                "password",
                "database",
                "connect_timeout",
                "sslmode",
            )
            if getattr(self, k, None) is not None
        }

        uri = URL.create(
            self.drivername,
            self.user,
            self.password,
            self.host,
            self.port,
            self.database,
        ).render_as_string(hide_password=False)

        return PostgresClient(conn_args, self.autocommit, get_dagster_logger(), uri)
