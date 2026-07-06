"""Temporary manual test script for ParentChildChunker's parent/child linking."""

import logging
from pathlib import Path

from llama_index.core.schema import NodeRelationship

from src.chunker.parent_child import ParentChildChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

PROCESSED_DIR = Path("data/processed")


def main() -> None:
    markdown_files = sorted(PROCESSED_DIR.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(
            f"No cached markdown files found in '{PROCESSED_DIR}'. Run test_parser.py first."
        )

    source_file = markdown_files[0]
    print(f"Reading cached markdown from: {source_file}")
    markdown_text = source_file.read_text(encoding="utf-8")

    chunker = ParentChildChunker()
    nodes = chunker.create_nodes(markdown_text)

    parent_nodes = [n for n in nodes if NodeRelationship.CHILD in n.relationships]
    child_nodes = [n for n in nodes if NodeRelationship.PARENT in n.relationships]

    print(f"\nTotal nodes: {len(nodes)}")
    print(f"Parent nodes: {len(parent_nodes)}")
    print(f"Child nodes: {len(child_nodes)}")

    for child in child_nodes:
        parent_id = child.relationships[NodeRelationship.PARENT].node_id
        assert child.metadata.get("parent_id") == parent_id, "Child metadata parent_id mismatch."
        assert any(p.node_id == parent_id for p in parent_nodes), "Child references unknown parent."

    print("\nParent/child relationships verified successfully.")


if __name__ == "__main__":
    main()
