import spacy

_nlp = None  # False = model unavailable; None = not yet attempted


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            try:
                import en_core_web_sm
                _nlp = en_core_web_sm.load()
            except ImportError:
                _nlp = spacy.load("en_core_web_sm")
        except OSError:
            _nlp = False  # model not installed — degrade gracefully
    return _nlp


def load_model() -> None:
    _get_nlp()


def analyze_framing(text: str) -> dict:
    nlp = _get_nlp()
    if not nlp:
        return {"entity_counts": {}, "modal_count": 0, "agentless_passive_ratio": 0.0, "agentfull_passive_ratio": 0.0, "sentence_count": 0, "average_sentence_length": 0.0}

    doc = nlp(text)

    entity_counts: dict[str, int] = {}
    for ent in doc.ents:
        entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1

    modal_count = sum(1 for tok in doc if tok.tag_ == "MD")

    sentences = list(doc.sents)
    sentence_count = len(sentences)
    total_tokens = sum(1 for tok in doc if not tok.is_punct and not tok.is_space)
    average_sentence_length = round(total_tokens / sentence_count, 3) if sentence_count else 0.0
    
    agentless_passive_sentences = sum(
        1 for sent in sentences
        if any(tok.dep_ in ("nsubjpass", "csubjpass") for tok in sent)
        and not any(tok.dep_ == "agent" for tok in sent)
    )
    
    agentfull_passive_sentences =sum(
        1 for sent in sentences
        if any(tok.dep_ in ("nsubjpass", "csubjpass") for tok in sent)
        and any(tok.dep_ == "agent" for tok in sent)
    )
    
    agentless_passive_ratio = round(agentless_passive_sentences / sentence_count, 3) if sentence_count else 0.0
    agentfull_passive_ratio = round(agentfull_passive_sentences / sentence_count, 3) if sentence_count else 0.0
    

    return {
        "entity_counts": entity_counts,
        "modal_count": modal_count,
        "agentless_passive_ratio": agentless_passive_ratio,
        "agentfull_passive_ratio": agentfull_passive_ratio,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
    }
