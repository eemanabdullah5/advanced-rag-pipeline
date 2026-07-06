"""ParentChildChunker: hierarchical chunking with parent/child node linking."""

import logging
from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document, NodeRelationship, RelatedNodeInfo

logger = logging.getLogger(__name__)

PARENT_CHUNK_SIZE = 1024
CHILD_CHUNK_SIZE = 256
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_OVERLAP = 20


class ParentChildChunker:
    def __init__(self) -> None:
        try:
            self.parent_splitter: SentenceSplitter = SentenceSplitter(
                chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
            )
            self.child_splitter: SentenceSplitter = SentenceSplitter(
                chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
            )
            logger.info(
                "ParentChildChunker initialized (parent_chunk_size=%d, child_chunk_size=%d)",
                PARENT_CHUNK_SIZE,
                CHILD_CHUNK_SIZE,
            )
        except Exception:
            logger.exception("Failed to initialize ParentChildChunker")
            raise

    def create_nodes(self, markdown_text: str) -> List[BaseNode]:
        try:
            document = Document(text=markdown_text)

            parent_nodes: List[BaseNode] = self.parent_splitter.get_nodes_from_documents([document])
            logger.info("Generated %d parent node(s)", len(parent_nodes))

            all_nodes: List[BaseNode] = []
            total_child_count = 0

            for parent_node in parent_nodes:
                child_nodes: List[BaseNode] = self.child_splitter.get_nodes_from_documents([parent_node])

                for child_node in child_nodes:
                    child_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                        node_id=parent_node.node_id
                    )
                    child_node.metadata["parent_id"] = parent_node.node_id

                parent_node.relationships[NodeRelationship.CHILD] = [
                    RelatedNodeInfo(node_id=child_node.node_id) for child_node in child_nodes
                ]

                total_child_count += len(child_nodes)
                all_nodes.append(parent_node)
                all_nodes.extend(child_nodes)

            logger.info(
                "Chunking complete: %d parent node(s), %d child node(s)",
                len(parent_nodes),
                total_child_count,
            )
            return all_nodes
        except Exception:
            logger.exception("Failed to create parent/child nodes")
            raise
