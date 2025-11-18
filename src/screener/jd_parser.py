import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# load once and relax the limit
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5_000_000  # allow up to 5M chars

_WORD_RE = re.compile(r"[A-Za-z]+")

def _simple_tokens(text: str):
    """Very fast fallback tokenizer (no spaCy) for huge texts."""
    text = text.lower()
    toks = [t for t in _WORD_RE.findall(text) if t not in STOP_WORDS]
    return list(set(toks))  # unique

def parse_jd(text: str):
    """
    Return unique keywords from JD.
    Uses spaCy normally; falls back to regex tokenization for very long texts.
    """
    if text is None:
        return []

    text = text.strip()
    # If insanely large, skip spaCy pipeline to avoid memory spikes
    if len(text) > 300_000:
        return _simple_tokens(text)

    try:
        doc = nlp(text.lower())
        tokens = [t.lemma_ for t in doc if t.is_alpha and t.text not in STOP_WORDS]
        return list(set(tokens))
    except Exception:
        # any parsing failure -> fallback
        return _simple_tokens(text)
