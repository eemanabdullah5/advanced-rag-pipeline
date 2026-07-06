"""Temporary manual test script for the retrieval layer (vector store + reranker)."""

import logging
import random

from src.retriever.reranker import FlashRankReranker
from src.retriever.vector_store import EMBEDDING_DIM, QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

COLLECTION_NAME = "test_collection"


def fake_embedding(seed: int) -> list:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]


def main() -> None:
    parent_id = "parent-1"
    parent_text = (
        "This is the full parent chunk covering the broader context of the financial report, "
        "including revenue, expenses, and forward-looking guidance."
    )

    child_chunks = [
        {
            "node_id": "child-1",
            "text": "Revenue grew 20% year over year driven by strong subscription sales.",
            "embedding": fake_embedding(seed=1),
            "parent_id": parent_id,
            "parent_text": parent_text,
        },
        {
            "node_id": "child-2",
            "text": "Operating expenses increased due to higher headcount and R&D investment.",
            "embedding": fake_embedding(seed=2),
            "parent_id": parent_id,
            "parent_text": parent_text,
        },
    ]

    print("\n--- Step 1: Initialize in-memory Qdrant vector store ---")
    vector_store = QdrantVectorStore(prefer_local=True)
    vector_store.initialize_collection(COLLECTION_NAME)

    print("\n--- Step 2: Upsert child chunks (with parent metadata in payload) ---")
    vector_store.upsert_child_chunks(COLLECTION_NAME, child_chunks)

    print("\n--- Step 3: Run a mock vector search ---")
    query_embedding = fake_embedding(seed=1)  # close to child-1's embedding
    search_results = vector_store.search(COLLECTION_NAME, query_embedding, top_k=10)
    assert len(search_results) == 2, "Expected both child chunks to be returned."
    for result in search_results:
        assert result["parent_id"] == parent_id
        assert result["parent_text"] == parent_text
    print(f"Retrieved {len(search_results)} candidate(s) from vector search.")

    print("\n--- Step 4: Re-rank candidates against the query ---")
    reranker = FlashRankReranker()
    query = "How much did revenue grow?"
    reranked_results = reranker.rerank(query, search_results)

    assert len(reranked_results) <= 5
    assert reranked_results, "Expected at least one reranked result."
    top_result = reranked_results[0]
    assert "rerank_score" in top_result
    assert top_result["parent_id"] == parent_id

    print(f"\nTop reranked result (score={top_result['rerank_score']:.4f}):")
    print(f"  child_text: {top_result['child_text']}")
    print(f"  parent_id:  {top_result['parent_id']}")

    print("\nEnd-to-end retrieval workflow verified successfully.")


if __name__ == "__main__":
    main()
