"""FlashRankReranker: cross-encoder re-ranking of merged candidate chunks."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

TOP_N_RESULTS = 5
DEFAULT_CACHE_DIR = Path("data/models/flashrank_cache")


class FlashRankReranker:
    def __init__(self, cache_dir: str = str(DEFAULT_CACHE_DIR)) -> None:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self.ranker: Ranker = Ranker(cache_dir=cache_dir)
            logger.info("FlashRankReranker initialized with Ranker model (cache_dir=%s)", cache_dir)
        except Exception:
            logger.exception("Failed to initialize FlashRankReranker")
            raise

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank candidate chunks against the query, returning the top 5 highest-scoring results.

        Each item in `candidate_chunks` is expected to provide at least a "text" key
        (falls back to "parent_text" or "child_text" if present), plus any metadata to
        preserve (e.g. "parent_id", "child_node_id").
        """
        try:
            if not candidate_chunks:
                logger.warning("No candidate chunks provided to rerank for query: '%s'", query)
                return []

            passages = []
            for idx, chunk in enumerate(candidate_chunks):
                text = chunk.get("text") or chunk.get("child_text") or chunk.get("parent_text") or ""
                passages.append({"id": idx, "text": text, "meta": chunk})

            logger.info(
                "Reranking %d candidate chunk(s) for query: '%s'", len(passages), query
            )

            start_time = time.perf_counter()
            rerank_request = RerankRequest(query=query, passages=passages)
            ranked_results = self.ranker.rerank(rerank_request)
            elapsed = time.perf_counter() - start_time

            logger.info("Reranking completed in %.3f seconds", elapsed)

            top_results: List[Dict[str, Any]] = []
            for result in ranked_results[:TOP_N_RESULTS]:
                original_chunk = result["meta"]
                enriched_chunk = {**original_chunk, "rerank_score": result["score"]}
                top_results.append(enriched_chunk)
                logger.debug(
                    "Rerank result: score=%.4f parent_id=%s",
                    result["score"],
                    original_chunk.get("parent_id"),
                )

            logger.info(
                "Returning top %d reranked result(s) out of %d candidate(s)",
                len(top_results),
                len(candidate_chunks),
            )
            return top_results
        except Exception:
            logger.exception("Failed to rerank candidates for query: '%s'", query)
            raise
