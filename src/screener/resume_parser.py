# src/screener/resume_parser.py
"""
Lightweight resume parser for name / skills / education / experience.
This file purposely does NOT import Streamlit or spaCy so it's safe to import.
"""

import re
from typing import Dict, List

# simple skill lexicon (extend as needed)
SKILLS = {
    "python", "sql", "tensorflow", "pytorch", "aws", "docker", "kubernetes",
    "react", "javascript", "html", "css", "node", "java", "c++", "c#", "git",
    "nlp", "machine learning", "deep learning", "excel"
}

SECTION_HEADERS_RE = re.compile(r"\b(skills|experience|education|projects|certifications|summary|objective|profile|contact)\b", re.I)
YEARS_RE = re.compile(r"(\d{1,2})(?:\+)?\s*(?:years|yrs)\b", re.I)

DEGREE_PATTERNS = [
    r"\bb\.?tech\b", r"\bbachelor", r"\bmtech\b", r"\bmaster", r"\bmsc\b", r"\bms\b",
    r"\bmba\b", r"\bph\.?d\b", r"\bbsc\b", r"\bba\b"
]
INSTITUTION_WORDS = [r"\buniversity\b", r"\bcollege\b", r"\binstitute\b", r"\bschool\b"]


def _clean_snippet(s: str) -> str:
    s = s.strip()
    s = re.split(SECTION_HEADERS_RE, s)[0]
    s = s.strip(" .:-\n\r\t")
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _looks_like_name(s: str) -> bool:
    # short, alphabetic tokens, 1-4 words
    s = s.strip()
    if not s or len(s) > 120:
        return False
    tokens = [t for t in re.split(r"\s+", s) if t]
    if not (1 < len(tokens) <= 5):
        return False
    return all(re.match(r"^[A-Za-z\-']+$", t) for t in tokens[:4])


def extract_name(text: str) -> str:
    if not text:
        return "Unknown"
    # explicit "Name:" lines first
    m = re.search(r"(?m)^\s*name\s*[:\-]\s*(.+)$", text, re.I)
    if m:
        name = _clean_snippet(m.group(1))
        return " ".join(w.title() for w in re.split(r"\s+", name) if w) or "Unknown"
    # else first short, name-like line
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if re.search(r"@|\bhttps?://|\d{7,}", ln):
            continue
        if _looks_like_name(ln):
            return " ".join(w.title() for w in re.split(r"\s+", ln)[:4])
    # fallback: email localpart
    m2 = re.search(r"([A-Za-z0-9._%+\-]+)@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if m2:
        parts = re.split(r"[._\-+]+", m2.group(1))
        parts = [p for p in parts if len(p) > 1]
        if parts:
            return " ".join(p.title() for p in parts[:3])
    return "Unknown"


def parse_resume(text: str) -> Dict[str, List[str]]:
    """
    Return dict with keys:
      - name: str
      - skills: list[str]
      - education: list[str] (short cleaned snippets)
      - experience: list[str] (short strings like '4 years')
    """
    if not text:
        return {"name": "Unknown", "skills": [], "education": [], "experience": []}

    lines = [ln.strip() for ln in text.splitlines()]
    # normalize lines: collapse and remove empty lines at ends
    normalized = [re.sub(r"\s{2,}", " ", ln).strip() for ln in lines]

    # name
    name = extract_name(text)

    # skills - heuristic token matching
    low = text.lower()
    skills_found = set()
    for skill in SKILLS:
        if " " in skill:
            if skill in low:
                skills_found.add(skill)
        else:
            if re.search(rf"\b{re.escape(skill)}\b", low):
                skills_found.add(skill)

    # education extraction:
    edu_snips = []
    # 1) If there's an explicit "Education" header, collect next 1..3 lines until another header
    header_idx = None
    for i, ln in enumerate(normalized):
        if re.match(r"(?i)^\s*education\s*[:\-]?\s*$", ln):
            header_idx = i
            break
    if header_idx is not None:
        count = 0
        for j in range(header_idx + 1, min(header_idx + 6, len(normalized))):
            ln = normalized[j]
            if not ln:
                if count:
                    break
                else:
                    continue
            if re.search(SECTION_HEADERS_RE, ln):
                break
            cleaned = _clean_snippet(ln)
            if cleaned:
                edu_snips.append(cleaned[:160])
                count += 1
            if count >= 3:
                break

    # 2) fallback: scan lines for degree/institution keywords
    if not edu_snips:
        for ln in normalized:
            if not ln:
                continue
            lowln = ln.lower()
            found_degree_or_instit = any(re.search(pat, lowln) for pat in DEGREE_PATTERNS + INSTITUTION_WORDS)
            if found_degree_or_instit:
                edu_snips.append(_clean_snippet(ln)[:160])
            if len(edu_snips) >= 3:
                break

    # dedupe, keep order
    seen = set()
    edu_out = []
    for e in edu_snips:
        if e not in seen:
            edu_out.append(e)
            seen.add(e)

    # experience extraction - find "X years" patterns
    exp_out = []
    m = re.search(r"experience\s*[:\-]\s*(\d{1,2})\s*(?:years|yrs)\b", low)
    if m:
        exp_out.append(f"{m.group(1)} years")
    for mm in YEARS_RE.finditer(low):
        token = f"{mm.group(1)} years"
        if token not in exp_out:
            exp_out.append(token)
    exp_out = exp_out[:3]

    return {
        "name": name,
        "skills": sorted(skills_found),
        "education": edu_out,
        "experience": exp_out,
    }
