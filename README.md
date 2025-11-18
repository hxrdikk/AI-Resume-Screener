# AI-Powered Resume Screener (NLP + Semantic Similarity)

A minimal, end-to-end project that parses resumes (text/PDF/DOCX), embeds them with Sentence-BERT, and ranks them against a Job Description using cosine similarity. Includes a Streamlit UI.

## 🔧 Tech Stack
- Python 3.10+
- sentence-transformers (`all-MiniLM-L6-v2` by default)
- spaCy (optional NER)
- PyMuPDF / pdfminer.six and python-docx (optional parsers for PDF/DOCX)
- Streamlit for the UI

## 🧰 Setup
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) spaCy small English model
python -m spacy download en_core_web_sm
```

## ▶️ CLI Usage
```bash
# Rank resumes in data/resumes against a JD in data/jds/jd1.txt
python -m screener.cli --jd data/jds/jd1.txt --resumes_dir data/resumes --top_k 10 --use_spacy
```

## 🖥️ Streamlit App
```bash
streamlit run src/app.py
```
- Upload a JD (text) and multiple resumes (txt/pdf/docx).  
- See ranked list with similarity scores.  
- Export results to CSV.

## 📂 Project Structure
```
ai-resume-screener/
├─ data/
│  ├─ resumes/        # place resumes here (txt/pdf/docx)
│  └─ jds/            # place job descriptions here
├─ src/
│  ├─ screener/
│  │  ├─ __init__.py
│  │  ├─ parser.py        # file loaders (txt/pdf/docx)
│  │  ├─ nlp.py           # cleaning, optional spaCy, embedding model
│  │  ├─ matcher.py       # cosine similarity + ranking
│  │  └─ cli.py           # command-line interface
│  └─ app.py              # Streamlit UI
├─ tests/
│  └─ test_matcher.py
├─ requirements.txt
├─ setup.cfg              # linters/formatters config
└─ README.md
```

## 🧪 Minimal Example
- We included 2 sample resumes and 1 JD to test the pipeline quickly.

## 📈 Notes
- For production with PDFs/DOCX, ensure `pymupdf`, `pdfminer.six`, and `python-docx` are installed.
- Consider using a domain-specific model (e.g., `all-mpnet-base-v2`) if accuracy needs to be higher.
- Add an ATS export format (CSV/JSON) as needed.
