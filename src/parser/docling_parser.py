"""DoclingParser: converts source documents to markdown with on-disk caching."""

import logging
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


class DoclingParser:
    def __init__(self) -> None:
        try:
            self.converter: DocumentConverter = DocumentConverter()
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("DoclingParser initialized. Cache directory: %s", PROCESSED_DIR.resolve())
        except Exception:
            logger.exception("Failed to initialize DoclingParser")
            raise

    def parse_or_load(self, file_path: str) -> str:
        source = Path(file_path)
        stem = source.stem
        cache_path = PROCESSED_DIR / f"{stem}.md"

        if cache_path.exists():
            try:
                logger.info("Cache hit for '%s' -> loading '%s'", file_path, cache_path)
                return cache_path.read_text(encoding="utf-8")
            except Exception:
                logger.exception("Failed to read cached file '%s'", cache_path)
                raise

        if not source.exists():
            logger.error("Source file '%s' does not exist", file_path)
            raise FileNotFoundError(f"Source file not found: {file_path}")

        file_size = source.stat().st_size
        logger.info("Cache miss for '%s' (%d bytes). Starting conversion.", file_path, file_size)

        start_time = time.perf_counter()
        try:
            result = self.converter.convert(str(source))
            markdown_content: str = result.document.export_to_markdown()
        except Exception:
            logger.exception("Failed to convert document '%s'", file_path)
            raise
        elapsed = time.perf_counter() - start_time
        logger.info("Converted '%s' to markdown in %.2f seconds", file_path, elapsed)

        try:
            cache_path.write_text(markdown_content, encoding="utf-8")
            logger.info("Cached markdown output at '%s'", cache_path)
        except Exception:
            logger.exception("Failed to write cache file '%s'", cache_path)
            raise

        return markdown_content
