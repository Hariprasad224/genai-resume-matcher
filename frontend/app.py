import streamlit as st
import requests

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.markdown("<style>" + open("ui.css").read() + "</style>", unsafe_allow_html=True)

st.title("🧠 AI Resume–JD Matcher")
st.subheader("Powered by Gemini + RAG")

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    jd = st.text_area("Paste Job Description")

    # 🔹 NEW: Evaluation method selector
    evaluation_method = st.radio(
        "Select Evaluation Method",
        options=["TF-IDF", "GenAI"],
        horizontal=True,
        help="TF-IDF is fast & keyword-based. GenAI is semantic & context-aware."
    )

    if st.button("Analyze Resume"):
        if resume is None or not jd.strip():
            st.warning("Please upload a resume and paste a job description.")
        else:
            files = {"resume": resume}
            data = {
                "job_description": jd,
                "method": evaluation_method.lower()  # tf-idf → tfidf, genai → genai
            }

            with st.spinner("Analyzing resume..."):
                response = requests.post(
                    "http://localhost:8000/analyze",
                    files=files,
                    data=data
                ).json()

            st.success(f"Match Score: {response['match_score']}%")

            st.markdown("### 📌 Analysis")
            st.write(response["analysis"])

            st.caption(f"Evaluation Method Used: **{evaluation_method}**")

    st.markdown("</div>", unsafe_allow_html=True)
