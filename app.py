import streamlit as st
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer, util
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load pre-trained model
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Efficient and accurate model
    return SentenceTransformer(model_name)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return " ".join([para.text for para in doc.paragraphs])

# Score the CV against the job description
def calculate_score(cv_text, jd_text, model):
    cv_embedding = model.encode(cv_text, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    similarity = util.cos_sim(cv_embedding, jd_embedding).item()
    return similarity * 100  # Convert to percentage



# Streamlit app
st.title("AI Resume Scorer")
st.write("Upload a resume and a job description to get started.")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description")

if st.button("Score Resume") and resume_file and job_description.strip():
    # Extract text from resume
    if resume_file.type == "application/pdf":
        cv_text = extract_text_from_pdf(resume_file)
    elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        cv_text = extract_text_from_docx(resume_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    # Load model
    model = load_model()

    # Calculate score
    score = calculate_score(cv_text, job_description, model)
    st.success(f"Resume Score: {score:.2f}%")

   
