import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embeddings(text):
    print("Generating embeddings...")
    print("Text length:", len(text))
    print("Text preview:", text[:100])  # Print first 100 characters for debugging
    response = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return response["embedding"]

def analyze_resume(resume, jd):
    # model = genai.GenerativeModel("gemini-pro")
    # print(list(genai.list_models()))
    model = genai.GenerativeModel("models/gemini-flash-lite-latest")

    prompt = f"""
    Compare the following resume with the job description.

    Resume:
    {resume}

    Job Description:
    {jd}

    Give:
    1. Key strengths
    2. Missing skills
    3. Improvement suggestions
    """

    response = model.generate_content(prompt)

    return response.text
