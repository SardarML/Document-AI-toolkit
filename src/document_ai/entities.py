"""General-purpose spaCy entity recognition."""

from functools import lru_cache


@lru_cache(maxsize=4)
def _load(name: str):
    import spacy

    return spacy.load(name)


def extract_entities(text: str, model: str = "en_core_web_sm") -> list[dict]:
    """Return entity spans and labels; not a domain-specific extractor."""
    return [
        {"text": entity.text, "label": entity.label_}
        for entity in _load(model)(text).ents
    ]
