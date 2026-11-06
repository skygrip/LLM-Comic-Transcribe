"""Text cleaning and validation heuristics for comic transcription outputs."""

import re

BUILTIN_RETRY_PATTERNS = [
    # Catch copyright refusals / canned safety responses
    r"(?:not able to|unable to|cannot|can't) (?:produce|provide|generate|share|create) (?:a |the )?(?:full |complete )?(?:verbatim )?transcription",
    r"(?:verbatim transcription|copyrighted (?:material|text|content))",
    # Catch generic AI refusal openings
    r"^(?:I(?:'m| am)? (?:cannot|can't|am unable to|not able to)|Sorry, (?:I|as an AI)|As an AI model)",
    # Catch hallucinated 'no text' commentary
    r"^(?:No text (?:found|detected|visible|present)|There is no text)",
    # Catch explicit [REDACTED] or [UNREADABLE] canary tags to retry before accepting
    r"\[(?:REDACTED|UNREADABLE)\]",
]


def clean_transcription_text(text: str) -> str:
    """Strips reasoning traces, code fences, accidental prefixes, and normalizes formatting.

    - Removes <thought> and <think> blocks
    - Strips wrapping markdown code fences
    - Removes accidental speaker prefixes before tags (Scene, SFX, Caption)
    - Normalizes multi-line Scene blocks and untagged opening summaries
    - Collapses consecutive duplicate lines
    """
    if not text:
        return ""

    # Remove <thought>...</thought> and <think>...</think> blocks (Gemma and Qwen/DeepSeek formats)
    cleaned = re.sub(r"<(thought|think)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove any unclosed/trailing markers
    cleaned = re.sub(r"</?(thought|think)>", "", cleaned, flags=re.IGNORECASE)

    # Repeatedly strip leading chat preambles and enclosing markdown code fences
    for _ in range(2):
        cleaned = re.sub(
            r"^(?:Here (?:is|are) (?:the )?(?:transcription|transcribed text|text)[^\n]*:?\s*\n*)",
            "",
            cleaned.strip(),
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^```(?:markdown|text)?\s*\n*", "", cleaned.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\n*```\s*$", "", cleaned.strip())

    # Strip accidental [Speaker]: prefixes before standalone (Scene: ...), (SFX: ...), or (Caption: ...) tags
    cleaned = re.sub(
        r"^\[[^\]]+\]:\s*(\((?:Scene|SFX|Caption):.*?\))\s*$",
        r"\1",
        cleaned,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Strip markdown asterisks inside (SFX: *...*) -> (SFX: ...)
    cleaned = re.sub(r"\(SFX:\s*\*+([^*]+?)\*+\s*\)", r"(SFX: \1)", cleaned, flags=re.IGNORECASE)

    # Collapse any multi-line (Scene: ...) block into a single line before splitting lines
    cleaned = re.sub(
        r"\(Scene:\s*(.*?)\)",
        lambda m: f"(Scene: {' '.join(m.group(1).split())})",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Collapse consecutive duplicate non-empty lines (case-insensitive comparison)
    deduped_lines = []
    prev_norm = None
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        norm = stripped.upper()
        if norm != prev_norm:
            deduped_lines.append(stripped)
            prev_norm = norm

    if not deduped_lines:
        return ""

    # If the model emitted untagged visual summary lines before the first [Speaker]: or (Tag: ...) line,
    # normalize them into a single (Scene: ...) line.
    first_tag_idx = next(
        (i for i, ln in enumerate(deduped_lines) if ln.startswith(("[", "("))),
        len(deduped_lines),
    )
    if first_tag_idx > 0:
        summary_text = " ".join(deduped_lines[:first_tag_idx])
        scene_line = f"(Scene: {summary_text})"
        deduped_lines = [scene_line, *deduped_lines[first_tag_idx:]]

    # Separate an opening (Scene: ...) line from subsequent transcription lines with a blank line
    if len(deduped_lines) > 1 and deduped_lines[0].lower().startswith("(scene:"):
        return f"{deduped_lines[0]}\n\n" + "\n".join(deduped_lines[1:])

    return "\n".join(deduped_lines)


def should_retry_transcription(
    cleaned_text: str,
    raw_text: str = "",
    custom_regex: str | None = None,
) -> tuple[bool, str]:
    """Evaluates whether a transcription result is invalid and warrants a retry.

    Returns:
        (should_retry: bool, reason: str)
    """
    # Check 1: Zero / empty content after cleaning
    if not cleaned_text or not cleaned_text.strip():
        return True, "Empty content (0 characters returned or stripped by cleaner)"

    # Check 2: Custom user-supplied regex pattern (from CLI --retry-regex or code extension)
    if custom_regex and re.search(custom_regex, cleaned_text, re.IGNORECASE):
        return True, f"Matched custom retry regex: {custom_regex!r}"

    # Check 3: Built-in failure / refusal / meta-commentary patterns
    for pattern in BUILTIN_RETRY_PATTERNS:
        if re.search(pattern, cleaned_text, re.IGNORECASE):
            return True, f"Matched retry regex pattern: {pattern!r}"

    return False, ""
