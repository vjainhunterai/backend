from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import fitz  # PyMuPDF for PDF text extraction
import os

app = FastAPI()

# Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/llama-3-8b"
HF_API_KEY = "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"  # Store API Key as Environment Variable

@app.post("/analyze_candidate/")
async def analyze_candidate(resume: UploadFile = File(...)):
    """Upload and analyze resume using Hugging Face API"""

    try:
        # Check file type
        if not resume.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        # Read uploaded file
        pdf_bytes = await resume.read()

        # Check if file is empty
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Extract text using PyMuPDF (fitz)
        try:
            pdf_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
            resume_text = "\n".join([page.get_text("text") for page in pdf_reader])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid PDF file.")

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")

        # Call Hugging Face API
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": f"Analyze this resume: {resume_text}"}
        )

        return response.json()

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
