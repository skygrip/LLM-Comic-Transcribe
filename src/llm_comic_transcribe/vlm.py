"""Vision-Language Model (VLM) API client and image optimization utilities."""

import base64
import io
import time
from typing import Any

import httpx
from PIL import Image
from tqdm import tqdm

from .cleaner import clean_transcription_text, should_retry_transcription


def get_image_mime(image_bytes: bytes) -> str:
    """Helper to detect image MIME type from magic header bytes."""
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
        return "image/webp"
    return "image/jpeg"


def optimize_image_for_vlm(image_bytes: bytes, max_dim: int = 2500) -> tuple[bytes, str]:
    """Compresses and constrains image dimensions to prevent oversized base64 payloads
    and server cache limit warnings, while maintaining high OCR fidelity.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        # Skip re-encoding if dimensions already fit and file size is modest (<500KB)
        if max(w, h) <= max_dim and len(image_bytes) < 500_000:
            return image_bytes, get_image_mime(image_bytes)

        if max(w, h) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        out = io.BytesIO()
        # Use quality 90 for history thumbnails to keep payload compact, 95 for full-res target
        quality = 90 if max_dim <= 1024 else 95
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, get_image_mime(image_bytes)


def transcribe_page_with_vlm(
    client: httpx.Client,
    api_url: str,
    model_name: str,
    image_bytes: bytes,
    system_prompt: str,
    history: list[tuple[bytes, str]] | None = None,
    max_image_size: int = 2500,
    history_image_size: int = 768,
    temperature: float | None = None,
    max_tokens: int = 4096,
    frequency_penalty: float | None = 0.2,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    reasoning_budget: int | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    max_retries: int = 3,
    retry_regex: str | None = None,
    page_num: int | None = None,
    timeout: float = 120.0,
    headers: dict[str, str] | None = None,
    quiet: bool = False,
) -> str:
    """Calls OpenAI-compatible /v1/chat/completions with image-only user payload and rolling history."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # Prepend rolling conversation history (prior pages as completed user/assistant turns)
    if history:
        for prev_bytes, prev_text in history:
            opt_prev_bytes, prev_mime = optimize_image_for_vlm(prev_bytes, max_dim=history_image_size)
            prev_b64 = base64.b64encode(opt_prev_bytes).decode("utf-8")
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{prev_mime};base64,{prev_b64}"},
                    },
                ],
            })
            messages.append({
                "role": "assistant",
                "content": prev_text,
            })

    # Current target page (pure image)
    opt_curr_bytes, curr_mime = optimize_image_for_vlm(image_bytes, max_dim=max_image_size)
    curr_b64 = base64.b64encode(opt_curr_bytes).decode("utf-8")
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{curr_mime};base64,{curr_b64}"},
            },
        ],
    })

    # Build request payload
    payload: dict[str, Any] = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if temperature is not None:
        payload["temperature"] = temperature

    # Configurable reasoning / thinking settings
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
        # If effort is explicitly set to none/off, ensure enable_thinking is synchronized
        if reasoning_effort.lower() in ("none", "off", "false", "disabled") and enable_thinking is None:
            payload["enable_thinking"] = False
    if reasoning_budget is not None:
        payload["reasoning_budget_tokens"] = reasoning_budget

    # Merge into chat_template_kwargs for Jinja chat templates (llama-server / llama-swap)
    merged_ctk: dict[str, Any] = {}
    if enable_thinking is not None:
        merged_ctk["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        merged_ctk["reasoning_effort"] = reasoning_effort
    if chat_template_kwargs:
        merged_ctk.update(chat_template_kwargs)
    if merged_ctk:
        payload["chat_template_kwargs"] = merged_ctk

    last_cleaned = ""
    last_reason = ""
    pg_str = f"Page {page_num}" if page_num is not None else "Page"

    def log_msg(msg: str):
        if quiet:
            print(msg, flush=True)
        else:
            tqdm.write(msg)

    for attempt in range(1, max(1, max_retries) + 1):
        try:
            response = client.post(api_url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            raw_content = data["choices"][0]["message"].get("content") or ""
            cleaned = clean_transcription_text(raw_content)

            retry_needed, reason = should_retry_transcription(
                cleaned, raw_text=raw_content, custom_regex=retry_regex
            )
            if not retry_needed:
                return cleaned

            last_cleaned = cleaned
            last_reason = reason
            if attempt < max_retries:
                log_msg(f"\n[Retry] {pg_str}: Attempt {attempt}/{max_retries} failed check ({reason}). Retrying...")
                time.sleep(1.0)
        except (httpx.HTTPStatusError, httpx.RequestError) as err:
            last_reason = f"Server/Network error ({err})"
            if attempt < max_retries:
                backoff = 2.0 * attempt
                log_msg(f"\n[Retry] {pg_str}: Attempt {attempt}/{max_retries} hit {last_reason}. Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
            else:
                log_msg(f"\n[Warning] {pg_str}: All {max_retries} attempts failed with {last_reason}.")
                raise

    log_msg(f"\n[Warning] {pg_str}: All {max_retries} attempts failed check ({last_reason}). Using last output.")
    return last_cleaned
