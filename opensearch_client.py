from opensearchpy import OpenSearch
from typing import List, Dict, Any

INDEX_NAME = "usd_assets"

def get_opensearch_client(host: str = "localhost", port: int = 9200) -> OpenSearch:
    """Returns initialized OpenSearch client."""
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )

def setup_opensearch_index(client: OpenSearch, index_name: str = INDEX_NAME):
    """Configures OpenSearch index with k-NN HNSW vector search mapping."""
    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil"
            }
        },
        "mappings": {
            "properties": {
                "asset_id": {"type": "keyword"},
                "usd_path": {"type": "keyword"},
                "thumbnail_url": {"type": "keyword"},
                "caption": {"type": "text"},
                "tags": {"type": "keyword"},
                "file_size_bytes": {"type": "long"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1152,
                    "method": {
                        "name": "hnsw",
                        "engine": "nmslib",
                        "space_type": "cosinesimil"
                    }
                }
            }
        }
    }
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=mapping)
        print(f"Created OpenSearch index '{index_name}' with 1536-d k-NN mapping.")

def hybrid_search(client: OpenSearch, query_vector: List[float], text_query: str = "", k: int = 12) -> Dict[str, Any]:
    """Executes hybrid vector + BM25 keyword query."""
    body = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": k
                            }
                        }
                    },
                    {
                        "match": {
                            "caption": {
                                "query": text_query,
                                "boost": 0.3
                            }
                        }
                    } if text_query else {"match_all": {}}
                ]
            }
        }
    }
    return client.search(index=INDEX_NAME, body=body)
