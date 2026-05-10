"""
FrameCheck — a small FastAPI service for analyzing news framing.

Roadmap:
  Day 1: scaffold + echo /analyze endpoint.
  Day 2: + Hugging Face sentiment (distilbert-base-uncased-finetuned-sst-2-english).
  Day 3: + SpaCy framing signals (NER, modals, passive voice).
  Day 4: + LLM frame summary (Ollama / Anthropic).
  Day 5: + HTML demo page at GET /demo.
  Deploy: switched sentiment to VADER (NLTK) — 3-class output, fits Render free tier.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import framing as framing_module
import llm as llm_module
import sentiment as sentiment_module


@asynccontextmanager
async def lifespan(app: FastAPI):
    sentiment_module.load_model()
    framing_module.load_model()
    yield


app = FastAPI(
    title="FrameCheck",
    description=(
        "Analyzes news headlines and articles for sentiment, linguistic "
        "framing signals, and potential bias indicators."
    ),
    version="0.6.0",
    lifespan=lifespan,
)

_DEMO_HTML = (Path(__file__).parent / "templates" / "demo.html").read_text(encoding="utf-8")


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Headline or article text to analyze.",
    )


class SentimentResult(BaseModel):
    label: str = Field(description="POSITIVE, NEGATIVE, or NEUTRAL")
    score: float = Field(description="Absolute VADER compound score, 0–1")


class FramingSignals(BaseModel):
    entity_counts: dict[str, int] = Field(description="Named-entity counts by type (PERSON, ORG, GPE, …)")
    modal_count: int = Field(description="Number of modal verbs (can/could/may/might/must/shall/should/will/would)")
    passive_ratio: float = Field(description="Fraction of sentences using passive voice, 0–1")
    sentence_count: int = Field(description="Total sentence count")


class FrameSummary(BaseModel):
    frame: str = Field(description="5-10 word label for the dominant narrative frame")
    loaded_language: list[str] = Field(description="Words or phrases with strong positive/negative connotation")
    rationale: str = Field(description="One-sentence explanation of the framing classification")


class AnalyzeResponse(BaseModel):
    received_text: str
    char_count: int
    word_count: int
    sentiment: SentimentResult
    framing: FramingSignals
    frame_summary: Optional[FrameSummary] = Field(
        default=None,
        description="LLM-generated frame analysis. None if LLM_BACKEND=none or Ollama unreachable.",
    )


@app.get("/")
def root():
    """Liveness check."""
    return {"status": "ok", "service": "FrameCheck", "version": "0.6.0"}


@app.get("/demo", response_class=HTMLResponse)
def demo():
    """Interactive demo page."""
    return _DEMO_HTML


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Returns text stats, sentiment (VADER), SpaCy framing signals, and an LLM frame summary.
    """
    sent = sentiment_module.analyze_sentiment(req.text)
    signals = framing_module.analyze_framing(req.text)
    summary = llm_module.frame_summary(req.text)
    return AnalyzeResponse(
        received_text=req.text,
        char_count=len(req.text),
        word_count=len(req.text.split()),
        sentiment=SentimentResult(**sent),
        framing=FramingSignals(**signals),
        frame_summary=FrameSummary(**summary) if summary else None,
    )
