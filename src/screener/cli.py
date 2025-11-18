"""
Command-line interface to rank resumes.
"""
from __future__ import annotations
import argparse, glob, os, pandas as pd
from .matcher import rank_resumes_against_jd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True, help="Path to JD text file")
    ap.add_argument("--resumes_dir", required=True, help="Directory of resumes (.txt/.pdf/.docx)")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--use_spacy", action="store_true")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out_csv", default="ranked.csv")
    args = ap.parse_args()

    with open(args.jd, "r", encoding="utf-8", errors="ignore") as f:
        jd_text = f.read()

    patterns = ["*.txt", "*.pdf", "*.docx"]
    resume_paths = []
    for pat in patterns:
        resume_paths.extend(glob.glob(os.path.join(args.resumes_dir, pat)))

    if not resume_paths:
        raise SystemExit("No resumes found. Put files into the resumes directory.")

    df = rank_resumes_against_jd(jd_text, resume_paths, use_spacy=args.use_spacy, model_name=args.model_name)
    df = df.head(args.top_k)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
