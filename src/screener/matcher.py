"""
Similarity and ranking utilities with semantic + skill matching.
"""
from __future__ import annotations

from typing import List, Dict, Any
import re
import pandas as pd
from .parser import load_file
from .nlp import clean_text, Embedder, extract_entities_spacy
# NOTE: do NOT import parse_resume at module level - import it lazily in the function
from .jd_parser import parse_jd, _simple_tokens
from sklearn.metrics.pairwise import cosine_similarity
import os


def extract_candidate_name(text: str, path: str) -> str:
    """
    Improved candidate name extractor:
    - Scans first 30 non-empty lines.
    - Accepts lines with 1–6 tokens if first word is capitalized.
    - Falls back to email local-part or filename.
    """
    lines = [ln.strip("\ufeff ").strip() for ln in text.splitlines() if ln.strip()]

    # 1) Scan first 30 lines
    for ln in lines[:30]:
        words = ln.split()
        if 1 <= len(words) <= 6:
            if words[0] and words[0][0].isupper():  # assume name if first word capitalized
                return ln.strip()

    # 2) Try email local-part
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if m:
        local = m.group(0).split("@")[0]
        guess = " ".join(re.split(r"[._+-]+", local)).strip()
        if guess:
            guess_tc = " ".join(w.capitalize() for w in guess.split() if w)
            if len(guess_tc.split()) <= 6:
                return guess_tc

    # 3) Fallback to filename
    fname = path.split("/")[-1]
    base = re.sub(r"\.(txt|pdf|docx)$", "", fname, flags=re.I)
    base = base.replace("_", " ").replace("-", " ").strip()
    return base or fname


# helper: detect if a string looks like a person name
def _looks_like_name(s: str) -> bool:
    if not s or len(s) > 120:
        return False
    # remove punctuation then check token pattern
    s2 = re.sub(r"[^A-Za-z\s\-']", " ", s).strip()
    tokens = [t for t in s2.split() if t]
    if not (1 < len(tokens) <= 5):
        return False
    # tokens should be alphabetic-ish and capitalized or all lower (we'll allow both)
    for t in tokens[:4]:
        if not re.match(r"^[A-Za-z\-']+$", t):
            return False
    return True


def _clean_education_list(edu_list: List[str], candidate_name: str | None = None) -> List[str]:
    out = []
    seen = set()
    for e in edu_list or []:
        if not isinstance(e, str):
            continue
        e_clean = e.strip()
        if not e_clean:
            continue
        # drop if it's obviously a name
        if _looks_like_name(e_clean):
            # if it exactly equals candidate name, drop
            if candidate_name and e_clean.strip().lower() == candidate_name.strip().lower():
                continue
            # otherwise drop as likely noise
            continue
        # remove extremely long lines and collapse whitespace
        e_clean = re.sub(r"\s{2,}", " ", e_clean)[:240].strip()
        if e_clean and e_clean not in seen:
            out.append(e_clean)
            seen.add(e_clean)
    return out


def _extract_education_block_via_regex(text: str) -> List[str]:
    """
    Fallback: quick regex to capture an 'Education:' block.
    Returns first meaningful line(s) following 'Education:'.
    """
    matches = re.search(r"(?is)education\s*[:\-]\s*(.+?)(?:\n\s*\n|$|\n(?:skills|experience|projects|certifications)\b)", text)
    if not matches:
        return []
    block = matches.group(1).strip()
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        # maybe it's a single-line block
        block_line = re.sub(r"\s{2,}", " ", block)
        return [block_line[:240]] if block_line else []
    # return up to 3 lines cleaned
    out = []
    for ln in lines[:3]:
        ln2 = re.sub(r"\s{2,}", " ", ln)
        if not _looks_like_name(ln2):
            out.append(ln2[:240])
    return out


def rank_resumes_against_jd(
    jd_text: str,
    resume_paths: List[str],
    use_spacy: bool = False,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> pd.DataFrame:
    """
    Compare resumes against a job description using embeddings + skill overlap.
    parse_resume is imported lazily here to avoid circular imports.
    """
    # Lazy import of parse_resume to avoid circular import issues
    try:
        from .resume_parser import parse_resume  # local import
    except Exception:
        # Fallback minimal parser if original isn't available for some reason
        def parse_resume(text: str) -> Dict[str, Any]:
            return {"skills": [], "education": [], "experience": []}

    embedder = Embedder(model_name=model_name)

    # Clean and parse job description
    jd_text = clean_text(jd_text)
    try:
        jd_keywords = parse_jd(jd_text)
    except Exception:
        jd_keywords = _simple_tokens(jd_text)

    resumes: List[str] = []
    for p in resume_paths:
        try:
            txt = clean_text(load_file(p))
        except Exception as e:
            txt = f"__ERROR__ {e}"
        resumes.append(txt)

    # Embeddings
    jd_vec = embedder.encode([jd_text])
    res_vecs = embedder.encode(resumes)
    sims = cosine_similarity(jd_vec, res_vecs).flatten()

    rows = []
    for path, text, score in zip(resume_paths, resumes, sims):
        # Candidate name detection
        candidate_name = extract_candidate_name(text, path)

        item: Dict[str, Any] = {
            "candidate_name": candidate_name,
            "similarity": float(score),
            "length_chars": len(text),
            # show first 5 lines of resume text for debugging (optional)
            "debug_preview": "\n".join(text.splitlines()[:5]) if not text.startswith("__ERROR__") else text,
        }

        if not text.startswith("__ERROR__"):
            # call parser (may be custom)
            try:
                resume_info = parse_resume(text) or {}
            except Exception:
                resume_info = {}

            # resume_info["skills"] expected to be a list
            skills_list = resume_info.get("skills") if isinstance(resume_info.get("skills"), list) else []
            matched_skills = sorted(set(skills_list).intersection(set(jd_keywords)))
            skill_overlap = len(matched_skills) / max(1, len(jd_keywords))
            final_score = 0.7 * score + 0.3 * skill_overlap

            # education processing - defensive cleanup
            raw_edu = resume_info.get("education") if isinstance(resume_info.get("education"), list) else []
            cleaned_edu = _clean_education_list(raw_edu, candidate_name)

            # fallback: try regex-based extraction if nothing valid
            if not cleaned_edu:
                cleaned_edu = _extract_education_block_via_regex(text)

            # final join for dataframe cell
            edu_cell = ", ".join(cleaned_edu) if cleaned_edu else ""

            # experience processing (expect list)
            exp_list = resume_info.get("experience") if isinstance(resume_info.get("experience"), list) else []
            exp_cell = ", ".join(exp_list[:3]) if exp_list else ""

            item.update({
                "final_score": round(final_score, 3),
                "skill_overlap": round(skill_overlap, 3),
                "matched_skills": ", ".join(matched_skills),
                "education": edu_cell,
                "experience": exp_cell,
            })

            if use_spacy:
                try:
                    ents = extract_entities_spacy(text)
                    item["ORGs"] = ", ".join(sorted(set(ents.get("ORG", [])[:5])))
                    item["PERSONs"] = ", ".join(sorted(set(ents.get("PERSON", [])[:3])))
                    item["DATEs"] = ", ".join(sorted(set(ents.get("DATE", [])[:3])))
                except Exception:
                    # if spacy entity extraction fails, skip it
                    pass

        rows.append(item)

    # Sort by final_score if available, else similarity
    if rows and "final_score" in rows[0]:
        sort_col = "final_score"
    else:
        sort_col = "similarity"

    df = pd.DataFrame(rows).sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df
