"""
File parsers for resumes and JDs.
Supports .txt (always), and optionally .pdf/.docx if libraries are available.
"""
from __future__ import annotations
from typing import Optional
import os

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception:
        # fallback: pdfminer
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path) or ""
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF {path}: {e}")

def read_docx(path: str) -> str:
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        raise RuntimeError(f"Failed to parse DOCX {path}: {e}")

def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")
