from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()



genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tfidf_match_score(resume_text, jd_text):
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    return {
        "method": "tfidf",
        "score": round(score * 100, 2),
        "explanation": "Score computed using TF-IDF cosine similarity"
    }


def genai_match_score(resume_text, jd_text):
    if not jd_text.strip():
        return {
            "method": "genai",
            "score": 0.0,
            "explanation": "Job description was empty"
        }

    prompt = build_prompt(resume_text, jd_text)
    
    model = genai.GenerativeModel("models/gemini-flash-lite-latest")

    response = model.generate_content(prompt)

    score = extract_score(response.text)

    return {
        "method": "genai",
        "score": round(score, 2),
        "explanation": "Score computed using GenAI semantic evaluation"
    }

    
def extract_score(text):
    match = re.search(r"\b(\d{1,3}(\.\d+)?)\b", text)
    if match:
        return float(match.group(1))
    return 0.0



def build_prompt(resume_text, jd_text):
    return f"""
    You are an ATS scoring engine.

    Compare the following Resume and Job Description.
    Evaluate how well the resume matches the job description.

    Scoring rules:
    - Score must be between 0 and 100
    - Consider skills, experience, tools, and role alignment
    - Ignore formatting issues
    - Output ONLY a single number (no text, no explanation)

    Resume:
    {resume_text}

    Job Description:
    {jd_text}

    Match Score:
    """


def calculate_match_score(resume_text, jd_text, method):
    if method == "tf-idf":
        return tfidf_match_score(resume_text, jd_text)

    elif method == "genai":
        return genai_match_score(resume_text, jd_text)

    else:
        raise ValueError("Invalid evaluation method selected")
