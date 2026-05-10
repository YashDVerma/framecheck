# FrameCheck — Project Plan & Context

> Drop this in front of any future Claude session (Code or Cowork) and it has full context to continue the work.

## Goal

Ship a small FastAPI service that analyzes news text for sentiment, framing signals, and potential bias — by **May 12, 2026, 11:55 PM ET**, to back an application for the **Student Assistant for Developer Role at the Consortium on Trust in Media and Technology (CTMT)**, UF College of Journalism and Communications.

The CTMT platform analyzes "framing, tone, and potential bias" in news content using NLP and LLMs and is being piloted in newsrooms. FrameCheck is a deliberate miniature of that. The point is to flip the cover letter from "willing to learn" to "just shipped something in this exact space."

## Stack & key decisions

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2
- **Sentiment:** Hugging Face Transformers — `distilbert-base-uncased-finetuned-sst-2-english` (small, ~250MB, fast)
- **Framing signals:** SpaCy + `en_core_web_sm` — NER counts, modal/hedging verb counts, passive-voice ratio
- **LLM frame summary:** Local **Ollama** for development; **Anthropic API** for the deployed version (Render free tier can't run Ollama). A thin `llm.py` adapter chooses backend by env var.
- **Hosting:** **Render free tier** for the live demo URL.
- **Repo:** github.com/YashDVerma/framecheck (private until Day 6, then public for the resume link)

## 7-day schedule

| Day | Date | Goal | Status |
| --- | --- | --- | --- |
| 1 | May 6 | FastAPI scaffold + echo `/analyze` endpoint | **DONE** |
| 2 | May 7 | Hugging Face sentiment pipeline | **DONE** |
| 3 | May 8 | SpaCy framing signals (NER, modals, passive voice) | **DONE** |
| 4 | May 9 | LLM frame summary (Ollama local + Anthropic adapter) | **DONE** |
| 5 | May 10 | Minimal HTML demo page served by FastAPI | **DONE** |
| 6 | May 11 | Deploy to Render free tier + polish README with screenshots | **DONE** |
| 7 | May 12 AM | Update resume + cover letter, submit application | pending |

Daily budget: ~2.5 hours. Total ~15 hours.

## Current state (end of Day 1)

Files in repo root:

- `main.py` — FastAPI app with `GET /` liveness check and `POST /analyze` echo (returns `received_text`, `char_count`, `word_count`). Validates non-empty input via `Field(min_length=1)` → 422 on empty.
- `requirements.txt` — `fastapi==0.115.0`, `uvicorn[standard]==0.32.0`, `pydantic==2.9.2`. Deliberately minimal; we add transformers / spacy / etc. on the days we use them so installs stay quick.
- `README.md` — public-facing project description with quickstart, roadmap, stack.
- `.gitignore` — Python standard plus model-cache directories (HF/SpaCy models are huge — never commit them).
- `docs/PROJECT_PLAN.md` — this file.

Day 1 verified working: `GET /` returns 200, `POST /analyze` with valid text returns the expected JSON, empty/missing input returns 422.

## Day 2 plan (next session)

1. `pip install transformers torch` (heavy — ~2GB for torch; expect several minutes)
2. Add a `sentiment.py` module that lazy-loads the HF pipeline on first use (cold-start friendly).
3. Extend `AnalyzeResponse` with a `sentiment` block: `{"label": "POSITIVE"|"NEGATIVE", "score": float}`.
4. Update `/analyze` to call the pipeline and include sentiment in the response.
5. Add a sample-input table to the README.

Watch out: SST-2 is binary (POSITIVE/NEGATIVE only — no neutral). For news, that's a real limitation worth calling out in the README. We'll keep the small/fast model for now and note in "what's next" that a 3-class or news-specific model would be a future improvement.

## Day 4 LLM-adapter shape (for reference)

```python
# llm.py
def frame_summary(text: str) -> dict:
    backend = os.getenv("LLM_BACKEND", "ollama")  # "ollama" | "anthropic"
    if backend == "ollama":
        return _ollama_call(text)
    return _anthropic_call(text)
```

Prompt should request a structured JSON response: `{"frame": str, "loaded_language": [str], "rationale": str}`.

## Yash — working preferences

- No emojis unless he uses them first.
- Don't ask 4 multiple-choice questions in a row. One or two real free-text questions, then a sensible default.
- Don't over-confirm. Reversible decisions: just pick and tell him.
- Be honest about what's actually impressive vs. fluff. Push back on weak ideas.
- Concrete code over prose explanations.
- Use a TodoList / task tracker for multi-step work.
- Always include a verification step (run tests, hit the endpoint, etc.) before declaring a day done.

## Background on Yash

M.S. Computer Science at the University of Florida (3.9 GPA, graduating Dec 2026). B.Tech CSE from Jaypee University (8.5 CGPA). Strongest in C# / Unity / XR. Solid Python + ML fundamentals (built a 2-layer MLP from scratch in NumPy → 99.3% test acc). Built a Pascal/Delphi → WebAssembly compiler solo using Java + ANTLR 4 + LLVM. One month of XR internship at Qvolv Technologies (Summer 2024). No prior shipped production work in SpaCy, Hugging Face, FastAPI, or LLM APIs — this project exists to fix that gap.

## CTMT role — what they actually want

Required: Python, NLP/sentiment libraries (SpaCy, PyTorch). Preferred: LLM/NLP workflows, Flask/FastAPI, computational linguistics interest. Responsibilities are maintaining and troubleshooting the backend of a language-analysis platform during pilot testing in newsrooms. Pay $16/hr, up to 20 hrs/week, summer term. Search panel chair: Janet Coats (janetcoats@ufl.edu).
