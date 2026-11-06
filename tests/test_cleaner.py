"""Unit tests for text cleaning and retry heuristics."""

from llm_comic_transcribe.cleaner import clean_transcription_text, should_retry_transcription


def test_clean_think_tags():
    raw = "<think>I need to analyze this panel carefully.</think>[Hero]: We made it!"
    assert clean_transcription_text(raw) == "[Hero]: We made it!"


def test_clean_thought_tags():
    raw = "<thought>Thinking about scene</thought>(Scene: The sun sets.)\n[Hero]: Goodbye."
    assert clean_transcription_text(raw) == "(Scene: The sun sets.)\n\n[Hero]: Goodbye."


def test_clean_code_fences():
    raw = "```markdown\n(Scene: City street.)\n[Batman]: I am vengeance.\n```"
    assert clean_transcription_text(raw) == "(Scene: City street.)\n\n[Batman]: I am vengeance."


def test_clean_conversational_intro():
    raw = "Here is the transcription for this page:\n[Hero]: Charge!\n(SFX: BOOM)"
    assert clean_transcription_text(raw) == "[Hero]: Charge!\n(SFX: BOOM)"


def test_clean_intro_combined_with_fences():
    raw = "Here is the transcription:\n```markdown\n(Scene: City street.)\n[Batman]: I am vengeance.\n```"
    assert clean_transcription_text(raw) == "(Scene: City street.)\n\n[Batman]: I am vengeance."


def test_clean_speaker_prefix_before_tags():
    raw = "[Narrator]: (Scene: Dark alleyway.)\n[Speaker]: (SFX: *CRASH*)"
    cleaned = clean_transcription_text(raw)
    assert "(Scene: Dark alleyway.)" in cleaned
    assert "(SFX: CRASH)" in cleaned
    assert "[Narrator]:" not in cleaned
    assert "[Speaker]:" not in cleaned


def test_collapse_duplicate_lines():
    raw = "[Soldier]: Retreat!\n[Soldier]: Retreat!\n[Soldier]: retreat!\n[Captain]: Hold the line!"
    cleaned = clean_transcription_text(raw)
    assert cleaned == "[Soldier]: Retreat!\n[Captain]: Hold the line!"


def test_normalize_untagged_summary():
    raw = "The team arrives at the glowing portal in the mountain.\n[Scientist]: It's active!"
    cleaned = clean_transcription_text(raw)
    assert cleaned.startswith("(Scene: The team arrives at the glowing portal in the mountain.)\n\n")
    assert "[Scientist]: It's active!" in cleaned


def test_should_retry_on_empty():
    retry, reason = should_retry_transcription("")
    assert retry is True
    assert "Empty" in reason


def test_should_retry_on_refusal():
    raw = "I am not able to produce a verbatim transcription of copyrighted material."
    retry, reason = should_retry_transcription(raw)
    assert retry is True
    assert "copyright" in reason or "refusal" in reason or "regex" in reason


def test_should_retry_on_custom_regex():
    raw = "[Speaker]: Something happened here."
    retry, reason = should_retry_transcription(raw, custom_regex=r"Something happened")
    assert retry is True
    assert "custom retry regex" in reason


def test_valid_transcription_no_retry():
    valid = "(Scene: Alex enters the lab.)\n\n[Alex]: Is anyone here?\n(SFX: BEEP)"
    retry, _ = should_retry_transcription(valid)
    assert retry is False
