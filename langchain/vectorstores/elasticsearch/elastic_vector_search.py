"""Wrapper around Elasticsearch vector database."""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.data_schema import ElasticDataSchemaBuilder
from langchain.vectorstores.data_schema.base import DataSchemaBuilder
from langchain.vectorstores.filters import ElasticFilter
from langchain.vectorstores.filters.base import VectorStoreFilter
from langchain.vectorstores.elasticsearch.elastic_conf import ElasticConf


def _script_query(query_vector: List[int], query_filter: VectorStoreFilter) -> Dict:
    return {
        "script_score": {
            "query": query_filter.to_query_string(),
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector},
            },
        }
    }


class ElasticVectorSearch(VectorStore):
    """Wrapper around Elasticsearch as a vector database.

    Example:
        .. code-block:: python

            from langchain import ElasticVectorSearch
            elastic_vector_search = ElasticVectorSearch(
                "http://localhost:9200",
                "embeddings",
                embedding_function
            )

    """

    def __init__(
            self,
            elastic_conf: ElasticConf,
            index_name: str,
            embedding_function: Callable
    ):
        """Initialize with necessary components."""
        try:
            import elasticsearch
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.elastic_conf = elastic_conf
        try:
            es_client = elastic_conf.elastic_client()  # noqa
        except ValueError as e:
            raise ValueError(
                f"Your elasticsearch client string is misformatted. Got error: {e} "
            )
        self.client = es_client

    @classmethod
    def setup_index(cls,
                    index_name: str,
                    data_schema_builder: DataSchemaBuilder,
                    **kwargs: Any):
        """Create index in vector store if needed"""
        elastic_conf: ElasticConf = kwargs.get("elastic_conf")

        try:
            import elasticsearch
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticearch`."
            )
        try:
            client = elastic_conf.elastic_client()
        except ValueError as e:
            raise ValueError(
                "Your elasticsearch client string is misformatted. " f"Got error: {e} "
            )

        # If the index already exists, we don't need to do anything
        client.indices.create(index=index_name, body=data_schema_builder.create_schema())

    def add_document_by_vector(self,
                               vectors: List[List[float]],
                               texts: Iterable[str],
                               metadatas: Optional[List[dict]] = None,
                               document_ids: Optional[List[str]] = None):
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            document_ids: the document ids to be specified instead of random uuid
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
            @param document_ids:
            @param metadatas:
            @param texts:
            @param vectors:
        """
        try:
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        requests = []
        ids = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            vector = vectors[i]
            _id = document_ids[i] if document_ids else str(uuid.uuid4())

            # Creating the request
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "vector": vector,
                "text": text,
                "metadata": metadata,
                "_id": _id,
            }
            ids.append(_id)
            requests.append(request)
        bulk(self.client, requests)
        # TODO: add option not to refresh
        self.client.indices.refresh(index=self.index_name)
        return ids

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            document_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            document_ids: the document ids to be specified instead of random uuid
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        try:
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        requests = []
        ids = []
        vectors = []
        for i, text in enumerate(texts):
            vectors.append(self.embedding_function(text))
        return self.add_document_by_vector(vectors, texts, metadatas, document_ids)

    def similarity_search(
            self, query: str, k: int = 4, query_filter: VectorStoreFilter = None, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query_filter: the filter to use before computing ANN algorithm
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function(query)
        return self.similarity_search_by_vector(embedding, k, query_filter, **kwargs)

    def similarity_search_by_vector(
            self, vector: List[int], k: int = 4, query_filter: VectorStoreFilter = None, **kwargs: Any
    ) -> List[Document]:
        """
        Return docs most similar to a given embedding vector

        Args:
            vector: embedding vector to look up documents similar to
            k: Number of Documents to return. Defaults to 4.
            query_filter: the filter to use before computing ANN algorithm
        """
        filtered_query = query_filter or ElasticFilter()
        script_query = _script_query(vector, filtered_query)
        response = self.client.search(index=self.index_name, query=script_query)
        scores = [hit["_score"] for hit in response["hits"]["hits"][:k]]
        hits = [hit["_source"] for hit in response["hits"]["hits"][:k]]
        documents = []
        for hit, score in zip(hits, scores):
            documents.append(
                Document(
                    page_content=hit["text"],
                    metadata=hit["metadata"],
                    similarity_score=score,
                )
            )
        return documents

    def similarity_search_by_id(
            self, doc_id: str, k: int = 4, query_filter: VectorStoreFilter = None, **kwargs: Any
    ) -> List[Document]:
        """
        Return docs most similar to a given embedding vector

        Args:
            doc_id: document ID to look up documents similar to
            k: Number of Documents to return. Defaults to 4.
            query_filter: the filter to use before computing ANN algorithm
        """

        # Fetching the document vector
        doc = self.client.get(index=self.index_name, id=doc_id, source_includes=["vector"])
        if doc["found"]:
            doc_vector = doc["_source"]["vector"]
            return self.similarity_search_by_vector(doc_vector, k, query_filter)
        else:
            raise ValueError(f"Document with id {doc_id} not found")

    def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20) -> List[Document]:
        raise NotImplementedError

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> ElasticVectorSearch:
        """Construct ElasticVectorSearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the Elasticsearch instance.
            3. Adds the documents to the newly created Elasticsearch index.

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import ElasticVectorSearch
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                elastic_vector_search = ElasticVectorSearch.from_texts(
                    texts,
                    embeddings,
                    elasticsearch_url="http://localhost:9200"
                )
        """
        elasticsearch_url = get_from_dict_or_env(
            kwargs, "elasticsearch_url", "ELASTICSEARCH_URL"
        )
        try:
            import elasticsearch
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticearch`."
            )

        index_name = uuid.uuid4().hex
        embeddings = embedding.embed_documents(texts)
        dim = len(embeddings[0])
        cls.setup_index(index_name, ElasticDataSchemaBuilder(dim), elastic_conf=ElasticConf(elasticsearch_url))
        elastic_vector = cls(ElasticConf(elasticsearch_url), index_name, embedding.embed_query)
        elastic_vector.add_texts(texts, metadatas, None)
        return elastic_vector
