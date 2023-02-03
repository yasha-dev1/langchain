"""Wrapper around Elasticsearch vector database."""
from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.data_schema import ElasticDataSchemaBuilder
from langchain.vectorstores.data_schema.base import DataSchemaBuilder
from langchain.vectorstores.filters.base import VectorStoreFilter
from langchain.vectorstores.elasticsearch.elastic_conf import ElasticConf


def _script_query(query_vector: List[int], query_filter: VectorStoreFilter) -> Dict:
    return {
        "script_score": {
            "query": {query_filter.to_query_string()},
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
        try:
            client = elasticsearch.Elasticsearch(elasticsearch_url)
        except ValueError as e:
            raise ValueError(
                "Your elasticsearch client string is misformatted. " f"Got error: {e} "
            )

        # If the index already exists, we don't need to do anything
        client.indices.create(index=index_name, ignore=400, body=data_schema_builder.create_schema())

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
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            _id = document_ids[i] if document_ids else str(uuid.uuid4())

            # Creating the request
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "vector": self.embedding_function(text),
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
        script_query = _script_query(embedding, query_filter)
        response = self.client.search(index=self.index_name, query=script_query)
        hits = [hit["_source"] for hit in response["hits"]["hits"][:k]]
        documents = [
            Document(page_content=hit["text"], metadata=hit["metadata"]) for hit in hits
        ]
        return documents

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
        try:
            client = elasticsearch.Elasticsearch(elasticsearch_url)
        except ValueError as e:
            raise ValueError(
                "Your elasticsearch client string is misformatted. " f"Got error: {e} "
            )

        index_name = uuid.uuid4().hex
        embeddings = embedding.embed_documents(texts)
        dim = len(embeddings[0])
        cls.setup_index(dim, index_name)

        requests = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            request = {
                "_op_type": "index",
                "_index": index_name,
                "vector": embeddings[i],
                "text": text,
                "metadata": metadata,
            }
            requests.append(request)
        bulk(client, requests)
        client.indices.refresh(index=index_name)
        return cls(ElasticConf(elasticsearch_url), index_name, embedding.embed_query)
