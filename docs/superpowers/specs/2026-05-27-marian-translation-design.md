# Translation Feature — MarianMT Redesign

**Date:** 2026-05-27  
**Project:** Video-to-Prompt WebUI  
**File:** `video_to_prompt_webui.py`

## Problem

argostranslate (current) produces unacceptable Thai output:
- Abnormal spacing between every syllable
- Wrong word choices ("หัวเรื่อง" for "subject")
- Quality too low for promotional/review content descriptions

## Goal

Replace argostranslate with Helsinki-NLP MarianMT via HuggingFace `transformers`.
Fully offline after one-time download. EN↔TH toggle with cache (no re-translate on switch).

## Dependencies

Add to `requirements.txt`:
```
transformers>=4.30.0
sentencepiece>=0.1.99
```

Install: `pip install transformers sentencepiece`

## Model Storage

- Base path: `<project_dir>/models/`
- EN→TH: `models/opus-mt-en-th/` (~300MB)
- TH→EN: `models/opus-mt-th-en/` (~300MB)
- HuggingFace model IDs: `Helsinki-NLP/opus-mt-en-th`, `Helsinki-NLP/opus-mt-th-en`
- Add `models/` to `.gitignore`

## Code Changes

### Remove (delete entirely)
- `_get_argos_pkg()`
- `_is_local_installed()` (argostranslate version)
- `download_local_models()` (argostranslate version)
- `translate_local()` (argostranslate version)
- `get_local_model_status()` (argostranslate version)
- All `from argostranslate import ...` references

### Add

**`_MARIAN_MODELS`** — model path map:
```python
_MARIAN_MODELS = {
    ("en", "th"): ("Helsinki-NLP/opus-mt-en-th", "models/opus-mt-en-th"),
    ("th", "en"): ("Helsinki-NLP/opus-mt-th-en", "models/opus-mt-th-en"),
}
_marian_cache: dict = {}  # in-memory model cache
```

**`_is_local_installed(from_code, to_code)`** — checks if `models/opus-mt-{from}-{to}/config.json` exists.

**`_load_marian(from_code, to_code)`** — loads MarianTokenizer + MarianMTModel from local path; caches in `_marian_cache`; raises if not installed.

**`download_local_models()`** — iterates `_MARIAN_MODELS`, calls `MarianTokenizer.from_pretrained(model_id, cache_dir=local_path)` and `MarianMTModel.from_pretrained(...)` for each pair not yet installed. Returns status string.

**`get_local_model_status()`** — returns `"✅ en→th  |  ✅ th→en"` style string based on `_is_local_installed()`.

**`translate_local(text, from_code, to_code)`** — tokenizes, runs model.generate(), decodes. Returns translated string or None on error.

### Keep unchanged
- `translate_text(text, lang)` — same interface, same cache logic
- `translate_cache` gr.State — EN/TH toggle cache unchanged
- UI buttons and event handlers — unchanged

## Data Flow

```
User clicks "TH ภาษาไทย"
  → on_translate(text, cache, "th")
    → cache["th"] exists? → return cached instantly
    → translate_text(text, "th")
      → translate_local(text, "en", "th")
        → _is_local_installed? No → return None
        → Yes → _load_marian("en","th") → tokenize → generate → decode
      → return (result, "✅ แปลเป็นไทย (offline)")
    → update cache {en: original, th: result}
    → return result
```

## Error Handling

- Model not downloaded → return original text + status "❌ โมเดลยังไม่ได้ดาวน์โหลด"
- Download fails mid-way → partial files remain; re-download overwrites safely
- Model load error → log + return None → translate_text returns original text

## .gitignore

Add:
```
models/
```

## Out of Scope

- argostranslate uninstall (leave in venv, just unused)
- GPU acceleration (CPU inference only)
- Languages other than EN↔TH
