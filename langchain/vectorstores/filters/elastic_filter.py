from typing import List, Any

from langchain.vectorstores.filters.base import VectorStoreFilter


class ElasticFilter(VectorStoreFilter):
    """Filter that uses ElasticSearch as a backend."""

    def __init__(self):
        self.bool_queries = []

    def add_filter_exact_match(self, field_name: str, field_value: List[Any]):
        should_clause = []
        for field_val in field_value:
            should_clause.append({"term": {field_name: field_val}})
        self.bool_queries.append({"should": should_clause})

    def to_query_string(self) -> str:
        return self.query_string


