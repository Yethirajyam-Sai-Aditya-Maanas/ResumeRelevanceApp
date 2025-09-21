# app.py — Automated Resume Relevance Checker (Hackathon MVP)

import streamlit as st
import PyPDF2
import docx
import re
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
import sqlite3
import pandas as pd

# ------------------- Helpers -------------------

def extract_text_from_pdf(file_obj):
    reader = PyPDF2.PdfReader(file_obj)
    texts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        texts.append(text)
    return "\n".join(texts)

def extract_text_from_docx(file_obj):
    doc = docx.Document(file_obj)
    texts = [p.text for p in doc.paragraphs]
    return "\n".join(texts)

def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

# Extract comma-separated skills from JD
SKILL_PATTERNS = [r"Must-?have:([^\n]+)", r"Good-?to-?have:([^\n]+)", r"Skills?:([^\n]+)"]

def extract_skills_from_jd(jd_text):
    jd_text = jd_text or ""
    skills = set()
    for pat in SKILL_PATTERNS:
        for m in re.finditer(pat, jd_text, flags=re.IGNORECASE):
            chunk = m.group(1)
            parts = re.split(r"[,;|]", chunk)
            for p in parts:
                s = p.strip()
                if s:
                    skills.add(s.lower())
    return list(skills)

def keyword_hard_match_score(resume_text, jd_skills):
    if not jd_skills:
        return 0.0, []
    resume = resume_text.lower()
    found = 0
    missing = []
    for skill in jd_skills:
        if skill in resume:
            found += 1
        else:
            ratio = fuzz.partial_ratio(skill, resume)
            if ratio > 80:
                found += 1
            else:
                missing.append(skill)
    return (found / len(jd_skills)) * 100.0, missing

def compute_scores(resume_text, jd_text, embedding_model):
    jd_skills = extract_skills_from_jd(jd_text)
    hard_score, missing = keyword_hard_match_score(resume_text, jd_skills)
    try:
        sim = embedding_model.similarity(resume_text, jd_text)
        soft_score = float(sim) * 100.0
    except Exception:
        soft_score = 0.0
    final = 0.6 * hard_score + 0.4 * soft_score
    verdict = 'Low'
    if final >= 75:
        verdict = 'High'
    elif final >= 50:
        verdict = 'Medium'
    return {
        'hard_score': hard_score,
        'soft_score': soft_score,
        'final_score': final,
        'verdict': verdict,
        'missing_skills': missing,
        'jd_skills': jd_skills
    }

def format_missing_skills(missing):
    if not missing:
        return "No explicit missing skills detected (based on JD keywords)."
    return "Missing skills: " + ", ".join(missing)

# ------------------- Embedding Model -------------------

class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # For offline demo, replace model_name with local folder path:
        # model_name = r"D:\SE Project\hackathon_model"
        self.model = SentenceTransformer(model_name)
    def embed(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)
    def similarity(self, text1, text2):
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        sim = util.pytorch_cos_sim(emb1, emb2)
        if sim.numel() == 1:
            return float(sim.item())
        return sim

# ------------------- Database -------------------

class DB:
    def __init__(self, path='evaluations.db'):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._ensure()
    def _ensure(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS evals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_preview TEXT,
            jd TEXT,
            score REAL,
            verdict TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        self.conn.commit()
    def insert_evaluation(self, data: dict):
        c = self.conn.cursor()
        c.execute('''INSERT INTO evals (resume_preview, jd, score, verdict) VALUES (?, ?, ?, ?)''',
                  (data.get('resume_preview'), data.get('jd'), data.get('score'), data.get('verdict')))
        self.conn.commit()
        return c.lastrowid
    def get_all(self):
        c = self.conn.cursor()
        c.execute('SELECT id, score, verdict, created_at FROM evals ORDER BY id DESC')
        rows = c.fetchall()
        return [dict(id=r[0], score=r[1], verdict=r[2], created_at=r[3]) for r in rows]

# ------------------- Streamlit App -------------------

st.set_page_config(page_title="Resume Relevance Checker", layout="wide")
st.title("Automated Resume Relevance Check — Hackathon MVP")

# Sidebar Demo JD
with st.sidebar:
    st.header("Demo Controls")
    demo_jd = st.checkbox("Use sample JD", value=True)
    if demo_jd:
        SAMPLE_JD = (
            "Role: Machine Learning Engineer\n"
            "Must-have: Python, PyTorch or TensorFlow, SQL, Machine Learning, Deep Learning\n"
            "Good-to-have: Docker, AWS, MLOps, HuggingFace, NLP\n"
            "Qualifications: B.Tech / M.Tech in CS or equivalent"
        )
    else:
        SAMPLE_JD = ""

# ------------------- Inputs -------------------

st.header("Inputs")
col1, col2 = st.columns([1,2])

with col1:
    uploaded_resume = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf","docx"])
    paste_resume = st.text_area("Or paste resume text (optional)", height=150)

with col2:
    jd_text = st.text_area("Paste Job Description (JD)", value=SAMPLE_JD, height=250)

# ------------------- Evaluation -------------------

if st.button("Evaluate"):
    if not (uploaded_resume or paste_resume) or not jd_text.strip():
        st.warning("Please provide a resume (upload or paste) and a JD.")
    else:
        with st.spinner("Parsing resume..."):
            resume_text = paste_resume.strip()
            if uploaded_resume and uploaded_resume.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_resume)
            elif uploaded_resume and uploaded_resume.type in (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ):
                resume_text = extract_text_from_docx(uploaded_resume)
            resume_text = clean_text(resume_text)
            jd_text_clean = clean_text(jd_text)

        st.subheader("Parsed Resume (preview)")
        st.write(resume_text[:2000] + ("..." if len(resume_text)>2000 else ""))

        db = DB("evaluations.db")
        model = EmbeddingModel()  # <- optionally replace with offline path

        with st.spinner("Computing scores..."):
            scores = compute_scores(resume_text, jd_text_clean, model)

        st.subheader("Result")
        st.metric("Relevance Score", f"{scores['final_score']:.1f}/100")
        st.write("*Verdict:*", scores['verdict'])
        st.write("*Hard match (keywords):*", f"{scores['hard_score']:.1f}/100")
        st.write("*Semantic match:*", f"{scores['soft_score']:.1f}/100")

        st.write("---")
        st.subheader("Missing / Suggested Skills")
        st.write(format_missing_skills(scores['missing_skills']))

        db.insert_evaluation({
            'resume_preview': resume_text[:1000],
            'jd': jd_text_clean[:1000],
            'score': scores['final_score'],
            'verdict': scores['verdict']
        })
        st.success("Evaluation saved to local DB (evaluations.db)")

# ------------------- Sidebar DB viewer -------------------

st.sidebar.header("Saved Evaluations")
if st.sidebar.button("Show saved evaluations"):
    db = DB("evaluations.db")
    rows = db.get_all()
    df = pd.DataFrame(rows)
    if df.empty:
        st.sidebar.info("No evaluations yet.")
    else:
        st.sidebar.dataframe(df)
