from fastapi import FastAPI, UploadFile, File
import requests
import fitz  # PyMuPDF for PDF text extraction

app = FastAPI()

# Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/llama-3-8b"
HF_API_KEY = "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"  # Store API Key as Environment Variable

@app.post("/analyze_candidate/")
async def analyze_candidate(resume: UploadFile = File(...)):
    """Upload and analyze resume using Hugging Face API"""

    try:
        # Read uploaded file as binary
        pdf_bytes = await resume.read()

        # Extract text using PyMuPDF (fitz)
        pdf_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
        resume_text = "\n".join([page.get_text("text") for page in pdf_reader])

        if not resume_text.strip():
            return {"error": "No text could be extracted from the PDF"}

        # Call Hugging Face API
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": f"Analyze this resume: {resume_text}"}
        )

        return response.json()

    except Exception as e:
        return {"error": str(e)}
