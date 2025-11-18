"""
NLP utilities: text cleaning, optional spaCy NER, and embedding model.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_entities_spacy(text: str) -> Dict[str, List[str]]:
    """
    Very light NER wrapper. Requires spaCy en_core_web_sm.
    Returns dict with keys: PERSON, ORG, GPE, DATE, etc.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        out = {}
        for ent in doc.ents:
            out.setdefault(ent.label_, []).append(ent.text)
        return out
    except Exception:
        # If spaCy or model not installed, return empty.
        return {}

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        # sentence-transformers auto-downloads models on first run
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
