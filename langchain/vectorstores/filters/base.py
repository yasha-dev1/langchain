"""Interface for vector store Filters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional


class VectorStoreFilter(ABC):
    """Interface for vector store Filters."""

    @abstractmethod
    def add_filter_exact_match(self, field_name: str, field_value: List[Any]):
        """Add a filter to the filter chain for the vector store to match exactly the value"""

    @abstractmethod
    def to_query_string(self, field_prefix: str = None) -> str:
        """Return the query string for the filter chain"""
