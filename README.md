<!-- ~welcome note -->
<p align="center">
    <img src="https://readme-typing-svg.herokuapp.com/?font=Righteous&size=35&center=true&vCenter=true&width=500&height=70&duration=4000&lines=Hello+there!;Welcome+to+my+Project!" />
</p>

<div style="margin-top:12px;"></div> 

<!-- ~about this project -->
<h3 align="left"> ✨ About this project:</h3>

<div style="margin-top:12px;"></div> 

- AI-Powered Resume Screener is an end-to-end NLP project that automatically parses, embeds, and ranks resumes against a given Job Description (JD) using Sentence-BERT embeddings and cosine similarity.

- The project supports TXT, PDF, and DOCX resumes, provides a CLI for batch screening, and includes an interactive Streamlit web app for recruiters and HR teams.

<!-- ~vision -->
<h3 align="left"> 💡 Vision:</h3>

~ To build an intelligent, fair, and scalable resume screening system that leverages AI and semantic understanding to help recruiters identify the right talent faster, reduce manual bias, and make hiring more data-driven and efficient.

<!-- ~features -->
<h3 align="left"> 🧩 Features:</h3>

- Resume Parsing – Supports TXT, PDF, and DOCX resume formats
- Semantic Matching – Uses Sentence-BERT for context-aware resume–JD matching
- Candidate Ranking – Ranks resumes using cosine similarity with clear scores
- Streamlit Dashboard – Interactive UI for uploading JDs and resumes
- CLI Screening Tool – Batch processing resumes directly from terminal
- Export Results – Download ranked candidates as CSV for easy sharing
- NLP Enhancement – Optional spaCy NER for skill and entity extraction
- Local & Secure – Runs fully locally, ensuring data privacy
- Demo Ready – Includes sample resumes and job descriptions for instant testing
- Extensible Design – Easy to add ATS integration, analytics, or advanced models

<!-- ~tech stack -->
<h3 align="left"> 🛠 Tech Stack:</h3>

- Python 3.10+ – Core language for building the screening pipeline
- Sentence-Transformers – Semantic embeddings using Sentence-BERT
- spaCy (Optional) – Named Entity Recognition and NLP preprocessing
- PyMuPDF / pdfminer.six – PDF resume parsing
- python-docx – DOCX resume parsing
- Scikit-learn – Cosine similarity and ranking logic
- Streamlit – Interactive web interface for resume screening
- CLI (Argparse) – Command-line batch processing tool
- Pytest – Unit testing for similarity and ranking modules
- Setup.cfg – Linting

<!-- ~project structure -->
<h3 align="left"> 🏗 Project Structure:</h3>

```
ai-resume-screener/
├─ data/
│  ├─ resumes/            # Resume files (TXT / PDF / DOCX)
│  └─ jds/                # Job descriptions
│
├─ src/
│  ├─ screener/
│  │  ├─ __init__.py
│  │  ├─ parser.py        # Resume & JD file parsers
│  │  ├─ nlp.py           # Text cleaning, embeddings, spaCy
│  │  ├─ matcher.py       # Similarity calculation & ranking
│  │  └─ cli.py           # Command-line interface
│  │
│  └─ app.py              # Streamlit web application
│
├─ tests/
│  └─ test_matcher.py     # Unit tests for ranking logic
│
├─ requirements.txt       # Project dependencies
├─ setup.cfg              # Linting & formatting configuration
└─ README.md              # Project documentation
```

<!-- ~installation & usage -->
<h3 align="left"> ⚙️ Installation & Usage:</h3>

1. Clone the Repository
```bash
git clone https://github.com/hxrdikk/AI-Resume-Screener.git
cd AI-Resume-Screener
```
2️. Create and activate virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```
3️. Install dependencies 
```bash
pip install -r requirements.txt
```

4️. Download spaCy English model
```bash
python -m spacy download en_core_web_sm
```
<!-- ~deployment -->
<h3 align="left"> 🚀 Deployment:</h3>

~ HealthNest is currently deployed on Vercel →
