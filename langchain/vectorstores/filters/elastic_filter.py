import json
from typing import List, Any

from langchain.vectorstores.filters.base import VectorStoreFilter


class ElasticFilter(VectorStoreFilter):
    """Filter that uses ElasticSearch as a backend."""

    def __init__(self):
        self.bool_queries = []

    def add_filter_exact_match(self, field_name: str, field_value: Any):
        self.bool_queries.append({"term": {"metadata." + field_name: field_value}})

    def to_query_string(self, field_prefix: str = None) -> dict:
        if len(self.bool_queries) == 0:
            return {"match_all": {}}
        else:
            return {
                "bool": {
                    "must": self.bool_queries
                }
            }
