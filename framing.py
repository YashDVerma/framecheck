import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def load_model() -> None:
    """Eagerly load the SpaCy model (call from lifespan to avoid cold-start lag)."""
    _get_nlp()


def analyze_framing(text: str) -> dict:
    doc = _get_nlp()(text)

    # Named-entity counts by type (PERSON, ORG, GPE, …)
    entity_counts: dict[str, int] = {}
    for ent in doc.ents:
        entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1

    # Modal verbs: tag MD covers can/could/may/might/must/shall/should/will/would
    modal_count = sum(1 for tok in doc if tok.tag_ == "MD")

    # Passive voice: sentences that contain a passive nominal subject
    sentences = list(doc.sents)
    sentence_count = len(sentences)
    passive_sentences = sum(
        1 for sent in sentences
        if any(tok.dep_ in ("nsubjpass", "csubjpass") for tok in sent)
    )
    passive_ratio = round(passive_sentences / sentence_count, 3) if sentence_count else 0.0

    return {
        "entity_counts": entity_counts,
        "modal_count": modal_count,
        "passive_ratio": passive_ratio,
        "sentence_count": sentence_count,
    }
