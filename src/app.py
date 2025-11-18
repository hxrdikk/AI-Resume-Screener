# src/app.py
import os
import re
import tempfile
from typing import List

import pandas as pd
import streamlit as st

# app uses the library functions
from screener.matcher import rank_resumes_against_jd
from screener.parser import load_file

# parse_resume is parser-only (no UI) — import normally
from screener.resume_parser import parse_resume

# ---------- Page config ----------
st.set_page_config(page_title="AI Resume Screener", layout="wide", initial_sidebar_state="expanded")

# ---------- Small CSS ----------
st.markdown(
    """
    <style>
    .block-container{max-width:1200px; padding:1.5rem 2rem;}
    h1 { font-size: 34px; margin-bottom: 0.1rem; }
    .subtitle { color: #9aa4b2; margin-top: 0; margin-bottom: 1.25rem; }
    .uploader-box { background-color: #0f1724; border-radius: 10px; padding: 10px; border: 1px solid rgba(255,255,255,0.03); }
    .download-btn { margin-top: 14px; }
    .small-muted { color:#9aa4b2; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("## 🔎 AI-Powered Resume Screener")
    st.markdown('<div class="subtitle">Upload a Job Description and resumes (TXT/PDF/DOCX) or a CSV dataset to get a ranked shortlist — semantic + skill matching.</div>', unsafe_allow_html=True)
with col2:
    st.markdown("**v1.0**")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
    use_spacy = st.checkbox("Extract extra entities with spaCy", value=False)
    top_k = st.number_input("Top K", min_value=1, value=10, step=1)
    st.markdown("---")
    st.markdown("Tips:", unsafe_allow_html=True)
    st.markdown("- Use a MiniLM model for speed.\n- Toggle spaCy to enable NER-based education extraction.", unsafe_allow_html=True)

# ---------- Uploaders ----------
jd_col, resume_col = st.columns([1, 1])
with jd_col:
    st.markdown("#### 📄 Upload Job Description (TXT/PDF/DOCX)")
    st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
    jd_file = st.file_uploader("", type=["txt", "pdf", "docx"], key="jd")
    st.markdown("</div>", unsafe_allow_html=True)

with resume_col:
    st.markdown("#### 📂 Upload Resumes (TXT/PDF/DOCX)")
    st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
    resume_files = st.file_uploader("", type=["txt", "pdf", "docx"], accept_multiple_files=True, key="resumes")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("#### 📑 Or upload a CSV (columns: type, text)")
csv_file = st.file_uploader("", type=["csv"], key="csv")

# ---------- helper: save uploaded file to temp and return text ----------
def file_to_text(uploaded) -> str:
    suffix = "." + uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        path = tmp.name
    return load_file(path)

# ---------- name cleaning helper (keeps UI-friendly names) ----------
def clean_name_raw(s: str) -> str:
    if not s:
        return "Unknown"
    s = re.split(r'[:\-–—|/]', s, 1)[0]
    s = re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z\s\-\'\.]', ' ', s)).strip()
    return " ".join(p.title() for p in s.split()[:5]) if s else "Unknown"

# ---------- compute action ----------
if st.button("🚀 Compute Rankings"):
    jd_text = None
    resume_paths: List[str] = []

    # Case CSV upload
    if csv_file is not None:
        try:
            df_csv = pd.read_csv(csv_file)
            jd_rows = df_csv[df_csv["type"].str.lower() == "jd"]
            if not jd_rows.empty:
                jd_text = jd_rows.iloc[0]["text"]
            resume_rows = df_csv[df_csv["type"].str.lower() == "resume"]
            for _, row in resume_rows.iterrows():
                text = row["text"]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                    tmp.write(text.encode("utf-8"))
                    tmp.flush()
                    resume_paths.append(tmp.name)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

    # Case regular files
    elif jd_file and resume_files:
        jd_text = file_to_text(jd_file)
        for f in resume_files:
            suffix = "." + f.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getbuffer())
                tmp.flush()
                resume_paths.append(tmp.name)

    if not jd_text or not resume_paths:
        st.warning("Please upload either a CSV or a JD + Resumes.")
    else:
        with st.spinner("Computing similarity and skill matches..."):
            # main ranking function lives in screener.matcher
            df = rank_resumes_against_jd(jd_text, resume_paths, use_spacy=use_spacy, model_name=model_name)

        # ---------- map temp filenames to nicer candidate names ----------
        try:
            name_map = {}
            for p in resume_paths:
                base = os.path.basename(p)
                try:
                    txt = load_file(p)
                except Exception:
                    txt = ""
                parsed = {}
                try:
                    parsed = parse_resume(txt) or {}
                except Exception:
                    parsed = {}

                # prefer parser name if present
                raw_name = parsed.get("name") if isinstance(parsed, dict) else None
                if raw_name and raw_name != "Unknown":
                    nice = clean_name_raw(raw_name)
                else:
                    # fallback: first plausible line or filename
                    # extract first non-empty line
                    first_line = ""
                    for ln in txt.splitlines():
                        ln = ln.strip()
                        if ln:
                            first_line = ln
                            break
                    if first_line:
                        nice = clean_name_raw(first_line)
                    else:
                        noext = os.path.splitext(base)[0]
                        nice = clean_name_raw(re.sub(r'[_\-\.\d]+', ' ', noext))

                name_map[base] = nice or base

            # replace values in df["candidate_name"] when present
            if "candidate_name" in df.columns:
                def _replace(val, idx):
                    if not isinstance(val, str):
                        return val
                    b = os.path.basename(val)
                    if b in name_map:
                        return name_map[b]
                    # try normalized matching
                    vnorm = re.sub(r'[^0-9a-z]', '', val.lower())
                    for k, v in name_map.items():
                        kn = re.sub(r'[^0-9a-z]', '', k.lower())
                        if kn == vnorm or kn in vnorm or vnorm in kn:
                            return v
                    # index fallback
                    if 0 <= idx < len(resume_paths):
                        return name_map.get(os.path.basename(resume_paths[idx]), val)
                    return val

                df["candidate_name"] = df.apply(lambda r: _replace(r["candidate_name"], r.name), axis=1)
        except Exception:
            # don't break UI for name-fixing issues
            pass

        # ---------- friendly headers ----------
        friendly = {
            "candidate_name": "Candidate Name",
            "final_score": "Final Score",
            "similarity": "Similarity",
            "skill_overlap": "Skill Overlap",
            "matched_skills": "Matched Skills",
            "education": "Education",
            "experience": "Experience",
        }
        df = df.rename(columns=friendly)

        # ---------- show results ----------
        st.subheader("📊 Ranked Candidates")
        cols = ["Candidate Name", "Final Score", "Similarity", "Skill Overlap", "Matched Skills", "Education", "Experience"]
        cols = [c for c in cols if c in df.columns]
        display_df = df[cols].head(top_k).copy()

        # show dataframe
        st.dataframe(display_df, use_container_width=True)

        # download CSV
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", data=csv_out, file_name="ranked_resumes.csv", mime="text/csv")

        # ---------- resume previews below the table ----------
        st.markdown("---")
        st.markdown("### Resume previews")
        for i, path in enumerate(resume_paths):
            base = os.path.basename(path)
            try:
                txt = load_file(path)
            except Exception:
                txt = ""
            parsed = {}
            try:
                parsed = parse_resume(txt) or {}
            except Exception:
                parsed = {}

            # matched skills from parsed vs JD (if present in df)
            matched = ""
            if "Matched Skills" in display_df.columns:
                # try to find row with this candidate (by name)
                candidate_name = name_map.get(base, base)
                row = df[df.get("Candidate Name", "") == candidate_name]
                if not row.empty and "Matched Skills" in row.columns:
                    matched = str(row.iloc[0].get("Matched Skills", ""))

            with st.expander(f"{name_map.get(base, base)} — preview", expanded=False):
                if matched:
                    # highlight matched skills in preview
                    preview_html = txt
                    # escape basic HTML then highlight words (quick)
                    safe_preview = preview_html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    for skill in (matched.split(",") if matched else []):
                        skill = skill.strip()
                        if not skill:
                            continue
                        # simple case-insensitive replace with mark
                        safe_preview = re.sub(fr"(?i)\b({re.escape(skill)})\b", r"<mark>\1</mark>", safe_preview)
                    st.markdown(f"<div style='white-space:pre-wrap'>{safe_preview}</div>", unsafe_allow_html=True)
                else:
                    st.code(txt[:10000])  # show up to 10k chars
