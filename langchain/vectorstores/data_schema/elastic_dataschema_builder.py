import json
from typing import List, Any

from langchain.vectorstores.data_schema.base import DataSchemaBuilder


class ElasticDataSchemaBuilder(DataSchemaBuilder):
    """Filter that uses ElasticSearch as a backend."""

    def __init__(self, metadata_mapping: dict = None, index_text: bool = False):
        self.metadata_mapping = metadata_mapping
        self.index_text = index_text

    def create_schema(self, dims: int) -> dict:
        base_mapping = {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": dims,
                }
            }
        }

        # add metadata mapping
        if self.metadata_mapping:
            base_mapping["properties"]["metadata"] = {
                "type": "object",
                "properties": self.metadata_mapping,
            }

        # whether to index text or not
        if self.index_text:
            base_mapping["properties"]["text"] = {
                "type": "text",
            }
        else:
            base_mapping["properties"]["text"] = {
                "type": "text",
                "enabled": False
            }

        return base_mapping
