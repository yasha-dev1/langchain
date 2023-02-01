"""
Elasticsearch Configuration to wrap all the needed information of elasticsearch connection and serve it for abstraction
"""

import typing as t


class ElasticConf:
    """Wrapper around Elasticsearch Configuration to transmit the needed information to the Elasticsearch Client"""

    def __init__(
            self,
            elasticsearch_url: str,
            # API
            api_key: t.Optional[t.Union[str, t.Tuple[str, str]]] = None,
            basic_auth: t.Optional[t.Union[str, t.Tuple[str, str]]] = None,
            bearer_auth: t.Optional[str] = None,
            opaque_id: t.Optional[str] = None
    ):
        self.elasticsearch_url = elasticsearch_url
        self.api_key = api_key
        self.basic_auth = basic_auth
        self.bearer_auth = bearer_auth
        self.opaque_id = opaque_id

    def elastic_client(self):
        """Converts the Elasticsearch Configuration to an Elasticsearch Client"""

        try:
            import elasticsearch
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        return elasticsearch.Elasticsearch(
            self.elasticsearch_url,
            api_key=self.api_key,
            basic_auth=self.basic_auth,
            bearer_auth=self.bearer_auth,
            opaque_id=self.opaque_id
        )
