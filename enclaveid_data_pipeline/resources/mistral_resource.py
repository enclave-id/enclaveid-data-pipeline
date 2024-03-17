from dagster import ConfigurableResource
from mistralai.client import MistralClient
from pydantic import Field


class MistralResource(ConfigurableResource):
    api_key: str = Field(description="The API key to use for authentication.")

    def get_client(self) -> MistralClient:
        return MistralClient(api_key=self.api_key)
