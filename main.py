
from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import fitz  # PyMuPDF for PDF text extraction
import os

app = FastAPI()

# Hugging Face API
#HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/llama-3-8b"
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
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

        # Debug: Check file size
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        print(f"Received PDF: {resume.filename}, Size: {len(pdf_bytes)} bytes")

        # Extract text using PyMuPDF (fitz)
        try:
            pdf_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
            resume_text = "\n".join([page.get_text("text") for page in pdf_reader])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")

        # Debug: Check extracted text
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")
        
        print(f"Extracted Resume Text: {resume_text[:500]}...")  # Print first 500 chars

        # Call Hugging Face API
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": f"Analyze this resume: {resume_text}"}
        )

        return response.json()

    except HTTPException as http_err:
        print(f"HTTP Error: {http_err.detail}")
        raise http_err

    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
