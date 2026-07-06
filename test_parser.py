"""Temporary manual test script for DoclingParser's caching behavior."""

import logging
from pathlib import Path

from src.parser.docling_parser import DoclingParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sample_file = RAW_DIR / "sample.html"
    if not sample_file.exists():
        sample_file.write_text(
            "<html><body><h1>Sample Document</h1>"
            "<p>This is a test document for DoclingParser.</p></body></html>",
            encoding="utf-8",
        )

    parser = DoclingParser()

    print("\n--- First call (expect cache MISS + conversion) ---")
    first_result = parser.parse_or_load(str(sample_file))
    print(first_result[:200])

    cache_path = PROCESSED_DIR / f"{sample_file.stem}.md"
    assert cache_path.exists(), "Expected cached markdown file to be created."

    print("\n--- Second call (expect cache HIT) ---")
    second_result = parser.parse_or_load(str(sample_file))
    print(second_result[:200])

    assert first_result == second_result, "Cached content should match freshly parsed content."
    print("\nCaching mechanism verified successfully.")


if __name__ == "__main__":
    main()
