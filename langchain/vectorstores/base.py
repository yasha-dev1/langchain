"""Interface for vector stores."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.data_schema.base import DataSchemaBuilder
from langchain.vectorstores.filters.base import VectorStoreFilter


class VectorStore(ABC):
    """Interface for vector stores."""

    @classmethod
    @abstractmethod
    def setup_index(cls,
                    index_name: str,
                    data_schema_builder: DataSchemaBuilder,
                    **kwargs: Any):
        """Create index in vector store if needed"""

    @abstractmethod
    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            document_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            document_ids: which ID to use to identify the given document in the vector store
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """

    @abstractmethod
    def similarity_search(
            self, query: str, k: int = 4, query_filter: VectorStoreFilter = None, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""

    @abstractmethod
    def similarity_search_by_id(
            self, doc_id: str, k: int = 4, query_filter: VectorStoreFilter = None, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to a document already saved with doc_id in vector store."""

    @abstractmethod
    def similarity_search_by_vector(
            self, vector: List[int], k: int = 4, query_filter: VectorStoreFilter = None, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to a given embedding vector from vector store."""

    def max_marginal_relevance_search(
            self, query: str, k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        raise NotImplementedError

    @classmethod
    def from_documents(
            cls,
            documents: List[Document],
            embedding: Embeddings,
            **kwargs: Any,
    ) -> VectorStore:
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    @abstractmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> VectorStore:
        """Return VectorStore initialized from texts and embeddings."""
