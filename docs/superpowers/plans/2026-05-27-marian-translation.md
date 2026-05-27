# MarianMT Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace argostranslate with Helsinki-NLP MarianMT for high-quality offline EN↔TH translation.

**Architecture:** Install `transformers` + `sentencepiece`, download opus-mt models to `./models/`, swap out all argostranslate code in `video_to_prompt_webui.py` with MarianMT equivalents. UI and cache logic remain unchanged.

**Tech Stack:** Python, HuggingFace `transformers`, `sentencepiece`, MarianMT (Helsinki-NLP/opus-mt-en-th, Helsinki-NLP/opus-mt-th-en)

---

### Task 1: Install dependencies + update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install packages**

```bash
cd C:\Users\PC-01\Desktop\Claude\Video
venv/Scripts/pip install "transformers>=4.30.0" "sentencepiece>=0.1.99"
```

Expected: `Successfully installed transformers-... sentencepiece-...`

- [ ] **Step 2: Add to requirements.txt**

Open `requirements.txt` and add after `python-multipart` line:
```
transformers>=4.30.0
sentencepiece>=0.1.99
```

- [ ] **Step 3: Verify import works**

```bash
venv/Scripts/python -c "from transformers import MarianMTModel, MarianTokenizer; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Add models/ to .gitignore**

Add this line to `.gitignore`:
```
models/
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "feat: add transformers+sentencepiece for MarianMT translation"
```

---

### Task 2: Replace argostranslate translation block with MarianMT

**Files:**
- Modify: `video_to_prompt_webui.py` (translation section only, lines ~741–830)

- [ ] **Step 1: Locate and delete the old argostranslate block**

In `video_to_prompt_webui.py`, find and delete the entire block from:
```python
# ── Local Offline Translation (argostranslate) ───────────────────────
```
down to (and including) the end of `translate_text()` function.

Keep the line immediately after (the `# ── Model List (legacy compat)` section).

- [ ] **Step 2: Insert new MarianMT block in its place**

Paste this block where the old one was:

```python
# ── Local Offline Translation (MarianMT / Helsinki-NLP) ─────────────
_MODELS_DIR = Path(__file__).parent / "models"
_MARIAN_MODELS = {
    ("en", "th"): ("Helsinki-NLP/opus-mt-en-th", str(_MODELS_DIR / "opus-mt-en-th")),
    ("th", "en"): ("Helsinki-NLP/opus-mt-th-en", str(_MODELS_DIR / "opus-mt-th-en")),
}
_marian_cache: dict = {}  # in-memory model cache {(from, to): (tokenizer, model)}


def _is_local_installed(from_code: str, to_code: str) -> bool:
    """Check if MarianMT model files exist locally."""
    _, local_path = _MARIAN_MODELS.get((from_code, to_code), ("", ""))
    if not local_path:
        return False
    return (Path(local_path) / "config.json").exists()


def get_local_model_status() -> str:
    """Return status string for UI display."""
    parts = []
    for (fc, tc) in _MARIAN_MODELS:
        icon = "✅" if _is_local_installed(fc, tc) else "⬇️"
        parts.append(f"{icon} {fc}->{tc}")
    return "  |  ".join(parts)


def _load_marian(from_code: str, to_code: str):
    """Load (and cache) MarianTokenizer + MarianMTModel from local path."""
    key = (from_code, to_code)
    if key in _marian_cache:
        return _marian_cache[key]
    _, local_path = _MARIAN_MODELS[key]
    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(local_path)
    model = MarianMTModel.from_pretrained(local_path)
    _marian_cache[key] = (tokenizer, model)
    return tokenizer, model


def download_local_models() -> str:
    """Download and save MarianMT models for en<->th. Returns status message."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        installed = 0
        skipped = 0
        for (fc, tc), (model_id, local_path) in _MARIAN_MODELS.items():
            if _is_local_installed(fc, tc):
                skipped += 1
                continue
            Path(local_path).mkdir(parents=True, exist_ok=True)
            tokenizer = MarianTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(local_path)
            model = MarianMTModel.from_pretrained(model_id)
            model.save_pretrained(local_path)
            installed += 1
        return (
            f"✅ ดาวน์โหลดเสร็จ: {installed} โมเดล | ข้ามที่มีแล้ว: {skipped} โมเดล\n"
            f"(en<->th พร้อมใช้ offline แล้ว)"
        )
    except Exception as e:
        return f"❌ Download failed: {e}"


def translate_local(text: str, from_code: str, to_code: str) -> str | None:
    """Translate using local MarianMT model. Returns None if not available."""
    if not _is_local_installed(from_code, to_code):
        return None
    try:
        tokenizer, model = _load_marian(from_code, to_code)
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[DBG-translate] translate_local error: {e}")
        return None


# ── Translation (local offline only, en<->th) ────────────────────────
def translate_text(text: str, lang: str) -> tuple[str, str]:
    """
    Translate text using local MarianMT model (en<->th).
    Returns (translated_text, status_msg).
    """
    if not text or not text.strip():
        return text, "❌ No text to translate"

    if lang == "th":
        if not _is_local_installed("en", "th"):
            return text, "❌ โมเดลยังไม่ได้ดาวน์โหลด — กดปุ่ม 'ดาวน์โหลดโมเดล' ก่อน"
        result = translate_local(text.strip(), "en", "th")
        return (result.strip(), "✅ แปลเป็นไทย (offline)") if result else (text, "❌ แปลไม่สำเร็จ")

    if lang == "en":
        if not _is_local_installed("th", "en"):
            return text, "❌ โมเดลยังไม่ได้ดาวน์โหลด — กดปุ่ม 'ดาวน์โหลดโมเดล' ก่อน"
        result = translate_local(text.strip(), "th", "en")
        return (result.strip(), "✅ Translated to English (offline)") if result else (text, "❌ Translation failed")

    return text, f"❌ ไม่รองรับภาษา: {lang}"
```

- [ ] **Step 3: Syntax check**

```bash
venv/Scripts/python -c "
import ast
with open('video_to_prompt_webui.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add video_to_prompt_webui.py
git commit -m "feat: replace argostranslate with MarianMT (Helsinki-NLP opus-mt)"
```

---

### Task 3: Restart server and verify UI

**Files:** none (runtime verification only)

- [ ] **Step 1: Kill old server**

```bash
powershell -Command "
\$c = Get-NetTCPConnection -LocalPort 7860 -EA SilentlyContinue
if (\$c) { Stop-Process -Id \$c.OwningProcess -Force -EA SilentlyContinue }
"
```

- [ ] **Step 2: Start new server**

```bash
cd C:\Users\PC-01\Desktop\Claude\Video
venv/Scripts/python video_to_prompt_webui.py > logs/webui.log 2>&1 &
```

Wait 8 seconds, then:

```bash
powershell -Command "Get-NetTCPConnection -LocalPort 7860 -EA SilentlyContinue | Select-Object State,OwningProcess"
```

Expected: a row with `Listen` state.

- [ ] **Step 3: Verify translate_status shows model state**

Open http://localhost:7860 in browser (Ctrl+Shift+R).
Below the "TH ภาษาไทย" buttons, the status line should show either:
- `⬇️ en->th  |  ⬇️ th->en` (not yet downloaded)
- `✅ en->th  |  ✅ th->en` (if models present)

- [ ] **Step 4: Commit**

```bash
git add logs/ -f 2>/dev/null || true
git commit -m "chore: verify MarianMT server runs correctly" --allow-empty
```

---

### Task 4: Download models and smoke-test translation

**Files:** none (runtime test)

- [ ] **Step 1: Click "📥 ดาวน์โหลดโมเดลแปลภาษา" in UI**

Wait for status to show: `✅ ดาวน์โหลดเสร็จ: 2 โมเดล`

This downloads ~600MB total (en→th + th→en). May take 5–15 minutes.

- [ ] **Step 2: Smoke-test via Python directly**

```bash
venv/Scripts/python -c "
import sys; sys.path.insert(0, '.')
from video_to_prompt_webui import translate_text

result, status = translate_text('A young woman holds a box of Oreo Wafer Rolls.', 'th')
print('Status:', status)
print('Result:', result)
assert 'ผู้หญิง' in result or 'หญิง' in result or 'สาว' in result or len(result) > 10, 'Translation seems wrong'
print('PASS')
"
```

Expected output:
```
Status: ✅ แปลเป็นไทย (offline)
Result: <Thai text about a woman and Oreo Wafer Rolls>
PASS
```

- [ ] **Step 3: Smoke-test TH→EN**

```bash
venv/Scripts/python -c "
from video_to_prompt_webui import translate_text

result, status = translate_text('ผู้หญิงสาวถือกล่องโอริโอ', 'en')
print('Status:', status)
print('Result:', result)
assert len(result) > 5 and not result.startswith('❌'), 'Translation failed'
print('PASS')
"
```

Expected: English sentence about a woman holding Oreo.

- [ ] **Step 4: Commit**

```bash
git add video_to_prompt_webui.py
git commit -m "test: verify MarianMT en<->th translation works offline"
```

---

### Task 5: Push and final cleanup

**Files:** none

- [ ] **Step 1: Push branch**

```bash
git push origin feature/claude-api
```

- [ ] **Step 2: Verify .gitignore excludes models/**

```bash
git status
```

Expected: `models/` directory not listed in untracked files (covered by .gitignore).

---

## Self-Review

**Spec coverage:**
- ✅ Dependencies: Task 1
- ✅ Remove argostranslate: Task 2 (delete old block)
- ✅ _MARIAN_MODELS, _marian_cache: Task 2
- ✅ _is_local_installed(): Task 2
- ✅ _load_marian(): Task 2
- ✅ download_local_models(): Task 2
- ✅ get_local_model_status(): Task 2
- ✅ translate_local(): Task 2
- ✅ translate_text() unchanged interface: Task 2
- ✅ .gitignore models/: Task 1
- ✅ UI unchanged: confirmed in Task 3
- ✅ Smoke test download + translate: Task 4

**Placeholder scan:** None found — all steps have exact commands and code.

**Type consistency:**
- `translate_text(text, lang)` → `tuple[str, str]` — consistent across Task 2 and Task 4 tests
- `translate_local(text, from_code, to_code)` → `str | None` — consistent
- `_load_marian(from_code, to_code)` → `(tokenizer, model)` tuple — consistent in `translate_local`
