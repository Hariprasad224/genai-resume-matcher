from fastapi import FastAPI, UploadFile, File, Form
from resume_parser import extract_text_from_pdf
from similarity import calculate_match_score
from gemini_utils import analyze_resume

app = FastAPI()

@app.post("/analyze")
async def analyze_resume_api(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    method: str = Form(...)
):
    print("Received analysis request with method:", method)
    print("Job Description length:", len(job_description))
    resume_text = extract_text_from_pdf(resume.file)
    match_score = calculate_match_score(resume_text, job_description, method)
    ai_feedback = analyze_resume(resume_text, job_description)

    return {
        "match_score": match_score["score"],
        "analysis": ai_feedback
    }
