> Note: `sentence-transformers` may download models during the first run, which is fine on Streamlit Cloud.

---

### **`README.md`** 


# Automated Resume Relevance Checker

This is a Streamlit-based web app that automatically evaluates how well a resume matches a given Job Description (JD).

## Features
- Upload PDF or DOCX resume or paste resume text
- Paste Job Description (JD) or use demo JD
- Computes:
  - Hard keyword match score
  - Semantic similarity score (via embeddings)
  - Final relevance score
  - Missing skills suggestion
- Saves evaluations in local SQLite DB (`evaluations.db`)

## Usage
1. Deploy on Streamlit Cloud or run locally:
   ```bash
   streamlit run streamlit_app.py

2. Upload your resume or paste text.
3. Paste the JD or use the demo JD.
4. Click "Evaluate" to get scores and suggestions.

## Requirements

See `requirements.txt` for all dependencies.

## License

For personal and educational use.

```

---

If you want, I can **also modify your code to automatically detect the embedding model offline** (so it won’t need downloading each time on Streamlit Cloud) and **finalize it for direct deployment**, so you just push the repo and click “Deploy.”  

Do you want me to do that next?
```
