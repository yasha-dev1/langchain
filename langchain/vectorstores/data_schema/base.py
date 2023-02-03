"""Interface for creating data schema in vector store if needed"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional


class DataSchemaBuilder(ABC):
    """Interface for creating data schema in vector store if needed"""

    @abstractmethod
    def create_schema(self) -> dict:
        """A data schema to be used in setup process of data schema generation of vector store"""
