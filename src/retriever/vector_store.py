"""QdrantVectorStore: manages the child-chunk vector index with parent-chunk payloads."""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768  # nomic-embed-text (Ollama) output dimension


class QdrantVectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333, prefer_local: bool = False) -> None:
        try:
            if prefer_local:
                self.client: QdrantClient = QdrantClient(":memory:")
                logger.info("QdrantVectorStore initialized in-memory (local, no server required)")
            else:
                self.client = QdrantClient(host=host, port=port)
                logger.info("QdrantVectorStore initialized, connecting to Qdrant at %s:%d", host, port)
        except Exception:
            logger.exception("Failed to initialize QdrantVectorStore")
            raise

    def initialize_collection(self, collection_name: str) -> None:
        try:
            existing_collections = [c.name for c in self.client.get_collections().collections]
            if collection_name in existing_collections:
                logger.info("Collection '%s' already exists. Skipping creation.", collection_name)
                return

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(
                "Initialized collection '%s' (dim=%d, distance=COSINE)",
                collection_name,
                EMBEDDING_DIM,
            )
        except Exception:
            logger.exception("Failed to initialize collection '%s'", collection_name)
            raise

    def upsert_child_chunks(
        self,
        collection_name: str,
        child_chunks: List[Dict[str, Any]],
    ) -> None:
        """Upsert child chunks, embedding their text and storing parent text/id in the payload.

        Each item in `child_chunks` is expected to provide:
          - "node_id": str
          - "text": str
          - "embedding": List[float] (length EMBEDDING_DIM)
          - "parent_id": str
          - "parent_text": str
        """
        try:
            start_time = time.perf_counter()
            points: List[qmodels.PointStruct] = []

            for chunk in child_chunks:
                embedding: List[float] = chunk["embedding"]
                if len(embedding) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}"
                    )

                child_node_id: str = chunk["node_id"]
                payload: Dict[str, Any] = {
                    "child_node_id": child_node_id,
                    "child_text": chunk["text"],
                    "parent_id": chunk["parent_id"],
                    "parent_text": chunk["parent_text"],
                }

                point_id = self._to_point_id(child_node_id)
                points.append(
                    qmodels.PointStruct(id=point_id, vector=embedding, payload=payload)
                )

                logger.debug(
                    "Prepared point for child_node_id=%s parent_id=%s payload_keys=%s",
                    child_node_id,
                    chunk["parent_id"],
                    list(payload.keys()),
                )

            self.client.upsert(collection_name=collection_name, points=points)

            elapsed = time.perf_counter() - start_time
            logger.info(
                "Upserted %d child chunk(s) into collection '%s' in %.3f seconds",
                len(points),
                collection_name,
                elapsed,
            )
        except Exception:
            logger.exception("Failed to upsert child chunks into collection '%s'", collection_name)
            raise

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        try:
            start_time = time.perf_counter()
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
            )
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Search in collection '%s' returned %d result(s) in %.3f seconds",
                collection_name,
                len(results),
                elapsed,
            )

            return [
                {
                    "score": hit.score,
                    "child_node_id": hit.payload.get("child_node_id") if hit.payload else None,
                    "child_text": hit.payload.get("child_text") if hit.payload else None,
                    "parent_id": hit.payload.get("parent_id") if hit.payload else None,
                    "parent_text": hit.payload.get("parent_text") if hit.payload else None,
                }
                for hit in results
            ]
        except Exception:
            logger.exception("Failed to search collection '%s'", collection_name)
            raise

    @staticmethod
    def _to_point_id(node_id: str) -> str:
        """Deterministically map an arbitrary node_id string to a valid Qdrant point UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))
