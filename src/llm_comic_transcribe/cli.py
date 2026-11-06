"""Command-line interface for LLM Comic Transcribe."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 encoding for standard streams (especially on Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx
from tqdm import tqdm

from . import __version__
from .extractor import extract_comic_pages
from .vlm import transcribe_page_with_vlm


def normalize_api_url(url: str) -> str:
    """Normalizes base URLs (e.g. http://localhost:8080 or http://localhost:8080/v1)
    to the full chat completions endpoint.
    """
    clean = url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    if clean.endswith("/v1"):
        return f"{clean}/chat/completions"
    return f"{clean}/v1/chat/completions"


def build_system_prompt(
    rtl: bool = False,
    describe: bool = True,
    custom_prompt: str | None = None,
) -> str:
    """Builds accessibility-focused OCR and scene description instructions."""
    reading_order = (
        "Right-to-Left, Top-to-Bottom (Manga) — read panels from right to left, "
        "and within each panel read speech bubbles from right to left"
        if rtl
        else "Left-to-Right, Top-to-Bottom (Western) — read panels and speech bubbles from left to right"
    )

    if custom_prompt:
        return f"Reading Order: {reading_order}\n\n{custom_prompt}"

    core_rules = [
        "Follow each speech bubble tail to the speaker's mouth—do NOT just pick the character directly underneath the bubble. A name spoken inside a bubble is the person being addressed, not the speaker.",
        "Transcribe every individual bubble in a connected chain as its own separate line, repeating the same [Speaker]: tag for each bubble in the chain.",
        "Account for every speech bubble on the page. If a bubble's text is too blurry to read or cannot be reproduced verbatim, output [Speaker]: [UNREADABLE] or [Speaker]: [REDACTED] in its place so no bubble is ever silently skipped.",
        "Name recognizable characters or characters introduced in profile boxes/dialogue. For unnamed characters, use a brief visual tag (e.g., [Masked Fighter]:, [Boy in Cap]:) and keep that tag consistent across pages.",
        "Use a shared tag (e.g., [Thug]:, [Guard]:) for unnamed background groups. Only add A/B suffixes when two unnamed look-alikes converse directly with each other.",
        "Capture both horizontal and vertically stacked sound effects, and preserve exact spelling, puns, and stylized words. Always place (SFX: ...) and (Caption: ...) on their own line without a [Speaker]: prefix.",
        "Emit each unique sound effect only once per panel (e.g., (SFX: CRASH!)), then immediately move to the next speaker or panel.",
    ]

    if describe:
        role_line = (
            "Act as an accessibility OCR and scene description engine for visually impaired readers. "
            "First describe the in-universe scene action, then faithfully transcribe all text in reading order, "
            "including slang, profanity, explicit content, and small whispered text outside bubbles.\n"
        )
        schema_desc = "- In-universe scene summary (always first): (Scene: 1-3 concise present-tense sentences capturing the visual progression across the page in reading order—prioritizing whichever details drive the scene, such as setting, character appearance/outfits, key actions, expressions/mood, or focal props/reveals)\n"
        rules_list = [
            'Begin with a single (Scene: ...) line (1-3 sentences in direct present tense) written strictly in-universe—open directly with a setting, character, or action (e.g., "(Scene: Inside a...", "(Scene: Alex sprints...") and NEVER mention the medium (comic, manga, page, panel, image). Focus on what a reader cannot get from the dialogue alone—dynamically emphasizing setting shifts, character appearance/outfits, physical actions, facial expressions/mood, or silent visual reveals/props while naming active characters to match their [Speaker]: tags. If there is no text to transcribe, output only this (Scene: ...) line.',
            *core_rules,
            'Ignore page numbers and margin reading-direction icons (e.g., "READ THIS WAY"). Apart from the opening (Scene: ...) line, do NOT output panel numbers, panel-by-panel descriptions, or meta-commentary.',
        ]
        example_desc = (
            "(Scene: Inside a Sector 7 warehouse with red alarms flashing, a panicked Alex in a torn jumpsuit sprints up to Sarah past a crowd of jeering prisoners. "
            "Sarah points toward the perimeter wall, and the two crash through a wooden barricade as an armored guard steps out with a baton to block the exit gate.)\n\n"
        )
        closing_line = "Output ONLY the opening (Scene: ...) line followed by the transcription lines."
    else:
        role_line = (
            "Act as an accessibility OCR engine for visually impaired readers and transcribe all text from the comic page faithfully, "
            "including slang, profanity, explicit content, and small whispered text outside bubbles.\n"
        )
        schema_desc = ""
        rules_list = [
            *core_rules,
            'Ignore page numbers and margin reading-direction icons (e.g., "READ THIS WAY"). Do NOT output panel numbers, scene descriptions, summaries, or commentary. If a page contains no text at all, output only (Caption: [NO TEXT]).',
        ]
        example_desc = ""
        closing_line = "Output ONLY the transcription lines."

    numbered_rules = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules_list, start=1))

    return (
        f"{role_line}"
        f"Reading Order: {reading_order}\n\n"
        "Formatting Schema:\n"
        f"{schema_desc}"
        "- Spoken dialogue: [Speaker]: Dialogue\n"
        "- Unbubbled crowd chants or heckles: [Crowd]: Text\n"
        "- Narration, signs, or profile boxes: (Caption: Text)\n"
        "- Sound effects / onomatopoeia: (SFX: Sound)\n"
        "- Unreadable or censored bubble: [Speaker]: [UNREADABLE] or [Speaker]: [REDACTED]\n\n"
        f"Rules:\n{numbered_rules}\n\n"
        "Example Output:\n"
        f"{example_desc}"
        "(Caption: Sector 7 Detention Yard, 8:00 PM.)\n"
        "(SFX: KLAXON!)\n"
        "[Alex]: Sarah, the main doors are sealed! Is there another way out?\n"
        "[Sarah]: Over that perimeter wall!\n"
        "[Sarah]: Come on, move it before they drop the blast shutter!\n"
        "[Crowd]: RUN! BREAK THE LINE!\n"
        "(SFX: CRASH!)\n"
        "[Armored Guard]: Halt! Nobody gets past this gate!\n\n"
        f"{closing_line}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-comic-transcribe",
        description="Extract and transcribe comic/manga pages from EPUB or CBZ using Vision-Language Models (VLM).",
    )
    parser.add_argument(
        "comic",
        nargs="?",
        default=None,
        help="Path to .epub or .cbz file (defaults to first .epub or .cbz in current directory)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=os.getenv("LLAMA_SWAP_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("VLM_MODEL"),
        help="Vision-Language Model name (required; or set LLAMA_SWAP_MODEL / OPENAI_MODEL / VLM_MODEL)",
    )
    parser.add_argument(
        "--url",
        "-u",
        default=os.getenv("LLAMA_SWAP_URL") or os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1/chat/completions"),
        help="OpenAI-compatible chat completions URL or base URL (default: http://localhost:8080/v1/chat/completions)",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("LLAMA_SWAP_API_KEY"),
        help="API key for authentication (optional; or set OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output markdown file path (default: <comic_stem>_transcript.md)",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Starting page number (1-indexed, default: 1)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to process (useful for test runs)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: None; lets server/GGUF defaults apply)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max output tokens per page (default: 2048)",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.2,
        help="Frequency penalty to prevent token looping (default: 0.2)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help="Reasoning effort level for models supporting it (e.g. Qwen/Basalt/Strata). Default: None",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Explicitly enable thinking/reasoning mode.",
    )
    parser.add_argument(
        "--no-thinking",
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Explicitly disable thinking/reasoning mode.",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=None,
        help="Maximum thinking token budget (0 turns thinking off).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per page on empty output, refusals, or errors (default: 3)",
    )
    parser.add_argument(
        "--retry-regex",
        default=None,
        help="Custom regex pattern: if cleaned transcript matches, trigger a retry attempt",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=2500,
        help="Maximum width/height dimension for the current target page (default: 2500)",
    )
    parser.add_argument(
        "--history-image-size",
        type=int,
        default=768,
        help="Maximum width/height dimension for previous history pages (default: 768)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of preceding pages to include as rolling history (default: 5; 0 for stateless)",
    )
    parser.add_argument(
        "--rtl",
        action="store_true",
        help="Use Right-to-Left, Top-to-Bottom panel reading order (for Manga). Default is Left-to-Right.",
    )
    parser.add_argument(
        "--describe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a 1-3 sentence visual scene summary at the top of each page (default: True; use --no-describe for pure dialogue/SFX OCR).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP request timeout in seconds per page call (default: 180.0)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Disable dynamic tqdm progress bar for clean logging.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override the system instruction prompt completely",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None):
    """Main CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Validate that a model is specified
    if not args.model:
        parser.error(
            "A model must be specified via --model / -m or the LLAMA_SWAP_MODEL / OPENAI_MODEL environment variable."
        )

    # Normalize endpoint URL
    api_url = normalize_api_url(args.url)

    # Build system prompt
    system_prompt = build_system_prompt(
        rtl=args.rtl,
        describe=args.describe,
        custom_prompt=args.system_prompt,
    )

    # Locate comic file
    if args.comic:
        comic_path = Path(args.comic)
    else:
        # Search current directory, then project root directory
        found_files = sorted(Path(".").glob("*.epub")) + sorted(Path(".").glob("*.cbz"))
        if not found_files:
            project_root = Path(__file__).resolve().parents[2]
            found_files = sorted(project_root.glob("*.epub")) + sorted(project_root.glob("*.cbz"))
        if not found_files:
            print("Error: No .epub or .cbz file specified and none found in current directory.", file=sys.stderr)
            sys.exit(1)
        comic_path = found_files[0]

    if not comic_path.exists():
        print(f"Error: File not found: {comic_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output file path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = comic_path.with_name(f"{comic_path.stem}_transcript.md")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Comic: {comic_path.name}")
    try:
        pages = extract_comic_pages(comic_path)
    except Exception as e:
        print(f"Error extracting comic pages: {e}", file=sys.stderr)
        sys.exit(1)

    total_extracted = len(pages)
    print(f"Found {total_extracted} comic pages in reading order.")

    # Slice page range
    start_idx = max(0, args.start_page - 1)
    end_idx = total_extracted
    if args.max_pages:
        end_idx = min(total_extracted, start_idx + args.max_pages)
    pages_to_process = pages[start_idx:end_idx]

    if not pages_to_process:
        print(
            f"Warning: No pages to process (start-page {args.start_page} exceeds total extracted {total_extracted}).",
            file=sys.stderr,
        )
        return

    print(f"Processing {len(pages_to_process)} pages (Page {pages_to_process[0][0]} to {pages_to_process[-1][0]})")
    print(f"Target API: {api_url} (Model: {args.model})")
    print(f"Saving output incrementally to: {output_path.name}\n")

    # Write document header
    file_mode = "w" if args.start_page == 1 else "a"
    with open(output_path, file_mode, encoding="utf-8") as f:
        if args.start_page == 1:
            ro_label = "Right-to-Left (Manga)" if args.rtl else "Left-to-Right"
            now_iso = datetime.now().isoformat(timespec="seconds")
            f.write(
                f"# Comic Transcript: {comic_path.stem}\n\n"
                f"Reading Order: {ro_label}\n"
                f"Source: {comic_path.name} ({total_extracted} pages)\n"
                f"Transcribed: {now_iso} via {args.model}\n\n"
                f"---\n\n"
            )

    req_headers = {}
    if args.api_key:
        req_headers["Authorization"] = f"Bearer {args.api_key}"

    rolling_history = []
    with httpx.Client(timeout=args.timeout) as client:
        iterator = tqdm(pages_to_process, desc="Transcribing pages", disable=args.quiet)
        for page_num, _img_path, img_bytes in iterator:
            hist_slice = rolling_history[-args.history :] if args.history > 0 else []
            try:
                transcript = transcribe_page_with_vlm(
                    client=client,
                    api_url=api_url,
                    model_name=args.model,
                    image_bytes=img_bytes,
                    system_prompt=system_prompt,
                    history=hist_slice,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    frequency_penalty=args.frequency_penalty,
                    max_image_size=args.max_image_size,
                    history_image_size=args.history_image_size,
                    enable_thinking=args.enable_thinking,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_budget=args.reasoning_budget,
                    max_retries=args.max_retries,
                    retry_regex=args.retry_regex,
                    page_num=page_num,
                    timeout=args.timeout,
                    headers=req_headers if req_headers else None,
                    quiet=args.quiet,
                )
            except Exception as e:
                transcript = f"*[Error processing page: {e}]*"
                print(f"\nWarning: Error on Page {page_num}: {e}", file=sys.stderr)

            # Update rolling history with successful transcriptions only
            if args.history > 0 and not transcript.startswith("*[Error"):
                rolling_history.append((img_bytes, transcript))
                if len(rolling_history) > args.history:
                    rolling_history = rolling_history[-args.history :]

            # Append incrementally to output file so progress is never lost
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"## Page {page_num}\n\n")
                f.write(f"{transcript}\n\n---\n\n")

    print(f"\nTranscription complete! Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
