# llm-comic-transcribe

A script to transcribe comics and manga (`.cbz` or `.epub`) into markdown using vision LLMs.

I mostly put this together to run against local models (via [llama-swap](https://github.com/mostlygeek/llama-swap), vLLM, Ollama, etc.), but anything with an OpenAI-compatible endpoint works fine.

What it does:
- Pulls comic images out of `.cbz` or `.epub` files in reading order.
- Generates a quick 1-2 sentence visual scene summary at the top of each page (for screen readers / accessibility).
- Transcribes dialogue, captions, and sound effects with speaker tags.
- Sends downscaled thumbnails of the last few pages along with the current one so the model doesn't forget who characters are from page to page.
- Saves progress incrementally to a markdown file as it goes, so you don't lose anything if it crashes or you hit Ctrl+C.

## Quick start

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
# Basic run (model is required)
uv run comic-transcribe -m <your-model> comic.cbz

# Or run the script directly:
uv run transcribe_comic.py -m <your-model> comic.epub
```

If you don't specify a comic path, it just grabs the first `.epub` or `.cbz` it finds in the current folder.

## Examples

Manga (reads panels right-to-left):
```bash
uv run comic-transcribe --rtl -m qwen2.5-vl-72b manga.cbz
```

Just OCR text (no scene descriptions):
```bash
uv run comic-transcribe --no-describe -m my-model comic.cbz
```

Test a couple pages first before running the whole thing:
```bash
uv run comic-transcribe -m my-model --start-page 1 --max-pages 3 comic.epub
```

Using OpenRouter or another remote API:
```bash
export OPENAI_API_KEY="sk-..."
uv run comic-transcribe \
  --url https://openrouter.ai/api/v1 \
  -m meta-llama/llama-3.2-11b-vision-instruct \
  comic.epub
```

## Useful flags

- `-m, --model`: The model name to pass to the completions endpoint (required, or set `LLAMA_SWAP_MODEL` / `OPENAI_MODEL`).
- `-u, --url`: OpenAI-compatible completions endpoint (defaults to `http://localhost:8080/v1/chat/completions`).
- `-k, --api-key`: Bearer token / API key if your endpoint needs auth (or set `OPENAI_API_KEY`).
- `--rtl`: Use right-to-left panel ordering for Japanese manga.
- `--no-describe`: Skip the visual scene description, just transcribe text.
- `-o, --output`: Output file path (defaults to `<comic_name>_transcript.md`).
- `--start-page` / `--max-pages`: Page range slicing.
- `--history`: Number of prior pages to send as rolling visual context (default: 5, set to 0 for stateless).
- `--timeout`: Per-page request timeout in seconds (default: 180).
- `-q, --quiet`: Turn off the tqdm progress bar.

## Example output

```markdown
## Page 1

(Scene: Inside a dark detention corridor with alarms flashing, Alex sprints past a broken security door while armored guards step out to block the exit.)

(Caption: Sector 7 - 20:00 Hours)
(SFX: KLAXON!)
[Alex]: Sarah, the main doors are sealed!
[Sarah]: Over the perimeter wall, hurry!
[Guard]: Halt!
(SFX: CRASH!)
```

## License

MIT
