import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_sia = None


def load_model() -> None:
    """Download VADER lexicon and initialize the analyzer (call from lifespan)."""
    global _sia
    nltk.download("vader_lexicon", quiet=True)
    _sia = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    scores = _sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return {"label": label, "score": round(abs(compound), 4)}
