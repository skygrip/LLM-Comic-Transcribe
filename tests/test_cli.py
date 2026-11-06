"""Unit tests for CLI parsing, prompt generation, and URL normalization."""

import pytest

from llm_comic_transcribe.cli import build_arg_parser, build_system_prompt, normalize_api_url


def test_model_required_failure(monkeypatch):
    # Ensure environment variables are not set
    monkeypatch.delenv("LLAMA_SWAP_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("VLM_MODEL", raising=False)

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.model is None

    # Verifying parser.error behaves as expected when model is missing
    with pytest.raises(SystemExit) as exc_info:
        from llm_comic_transcribe.cli import main
        main([])
    assert exc_info.value.code == 2


def test_model_supplied_via_cli(monkeypatch):
    monkeypatch.delenv("LLAMA_SWAP_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("VLM_MODEL", raising=False)

    parser = build_arg_parser()
    args = parser.parse_args(["--model", "my-vlm-model"])
    assert args.model == "my-vlm-model"

    args_short = parser.parse_args(["-m", "another-vlm"])
    assert args_short.model == "another-vlm"


def test_model_supplied_via_env(monkeypatch):
    monkeypatch.setenv("LLAMA_SWAP_MODEL", "env-model-name")
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.model == "env-model-name"


def test_normalize_api_url():
    assert normalize_api_url("http://localhost:8080") == "http://localhost:8080/v1/chat/completions"
    assert normalize_api_url("http://localhost:8080/") == "http://localhost:8080/v1/chat/completions"
    assert normalize_api_url("http://localhost:8080/v1") == "http://localhost:8080/v1/chat/completions"
    assert (
        normalize_api_url("http://localhost:8080/v1/chat/completions")
        == "http://localhost:8080/v1/chat/completions"
    )
    assert (
        normalize_api_url("https://openrouter.ai/api/v1")
        == "https://openrouter.ai/api/v1/chat/completions"
    )


def test_build_system_prompt_rtl_vs_ltr():
    ltr_prompt = build_system_prompt(rtl=False, describe=True)
    assert "Left-to-Right" in ltr_prompt
    assert "Scene:" in ltr_prompt

    rtl_prompt = build_system_prompt(rtl=True, describe=True)
    assert "Right-to-Left, Top-to-Bottom (Manga)" in rtl_prompt


def test_build_system_prompt_no_describe():
    prompt = build_system_prompt(rtl=False, describe=False)
    assert "Do NOT output panel numbers, scene descriptions" in prompt
    assert "Caption: [NO TEXT]" in prompt
