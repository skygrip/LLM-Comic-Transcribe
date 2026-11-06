# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx>=0.27.0",
#     "pillow>=10.0.0",
#     "tqdm>=4.66.0",
# ]
# ///

"""Standalone entrypoint for LLM Comic Transcribe."""

import sys
from pathlib import Path

# Add src/ to sys.path for direct script execution without installing wheel
_src_dir = Path(__file__).resolve().parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from llm_comic_transcribe import (
    clean_transcription_text,
    extract_comic_pages,
    get_image_mime,
    optimize_image_for_vlm,
    should_retry_transcription,
    transcribe_page_with_vlm,
)
from llm_comic_transcribe.cli import main

__all__ = [
    "clean_transcription_text",
    "extract_comic_pages",
    "get_image_mime",
    "main",
    "optimize_image_for_vlm",
    "should_retry_transcription",
    "transcribe_page_with_vlm",
]

if __name__ == "__main__":
    main()
