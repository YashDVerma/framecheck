# FrameCheck

A small FastAPI service that analyzes news text for sentiment, linguistic framing signals, and potential bias indicators. Built as a focused 7-day project to demonstrate practical NLP and LLM integration.

## What it does

Send a headline or article body to `POST /analyze`; get back:

- **Sentiment** — positive / negative / neutral, via a Hugging Face transformer (`distilbert-base-uncased-finetuned-sst-2-english`).
- **Framing signals** — derived with SpaCy: named-entity counts (who is mentioned, how often), modal-verb / hedging-language counts (signals of certainty), passive-voice ratio (a known framing indicator).
- **LLM frame summary** — a structured "what frame is this article using, and what loaded language did you find?" response.

## Status

**Day 6 of 7 — deployed.** Live demo: **https://framecheck.onrender.com/demo**

> Free-tier note: the service sleeps after 15 minutes of inactivity. First request after sleep takes ~30 s to wake up; subsequent requests are fast. Both ML models are pre-downloaded at build time so there is no extra model-load delay on wake-up.

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dev server (auto-reloads on file changes)
uvicorn main:app --reload
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The senator denied allegations, but critics remain skeptical."}'
```

You can also browse the auto-generated Swagger docs at <http://127.0.0.1:8000/docs>.

## Deploy to Render

`render.yaml` is included — Render detects it automatically when you connect the repo.

```bash
# 1. Push to GitHub
git init
git add main.py framing.py llm.py requirements.txt render.yaml templates/ docs/ README.md .gitignore
git commit -m "FrameCheck v0.5.0"
git remote add origin https://github.com/YashDVerma/framecheck.git
git push -u origin main

# 2. Create a new Web Service on render.com → connect the repo
#    Render picks up render.yaml automatically.

# 3. In the Render dashboard → Environment tab, add:
#    ANTHROPIC_API_KEY = <your key>
#    (LLM_BACKEND=anthropic is already set in render.yaml)
```

The build command in `render.yaml` pre-downloads both ML models (DistilBERT + `en_core_web_sm`) so cold-start wake-ups are fast — no model download on every restart.

## Roadmap

| Day | Goal |
| --- | --- |
| 1 | FastAPI scaffold + `/analyze` echo endpoint ✓ |
| 2 | Hugging Face sentiment pipeline ✓ |
| 3 | SpaCy framing signals (NER, modals, passive voice) ✓ |
| 4 | LLM frame summary (Ollama local / Anthropic API) ✓ |
| 5 | Minimal HTML demo page ✓ |
| 6 | Deploy to Render free tier ✓ |
| 7 | Documentation polish + screenshots |

## Stack

Python · FastAPI · Hugging Face Transformers · SpaCy · Ollama / Anthropic API · Render

## Author

Yash Deep Verma — M.S. Computer Science, University of Florida.
[GitHub](https://github.com/YashDVerma) · [LinkedIn](https://linkedin.com/in/yashdverma)
