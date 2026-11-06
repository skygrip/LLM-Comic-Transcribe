"""LLM Comic Transcribe: Extract and transcribe comic and manga pages using Vision-Language Models."""

from .cleaner import clean_transcription_text, should_retry_transcription
from .extractor import extract_comic_pages
from .vlm import get_image_mime, optimize_image_for_vlm, transcribe_page_with_vlm

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "clean_transcription_text",
    "extract_comic_pages",
    "get_image_mime",
    "optimize_image_for_vlm",
    "should_retry_transcription",
    "transcribe_page_with_vlm",
]
