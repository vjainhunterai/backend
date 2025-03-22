from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import fitz  # PyMuPDF for PDF text extraction
import os

app = FastAPI()

# Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HF_API_KEY = "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"  # Store API Key as Environment Variable

def query_huggingface(prompt, resume_text):
    """Helper function to query Hugging Face API with different prompts"""
    response = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": f"{prompt}\n\n{resume_text}"}
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail="Hugging Face API request failed.")

@app.post("/analyze_candidate/")
async def analyze_candidate(resume: UploadFile = File(...)):
    """Upload and analyze resume with multiple prompts"""
    try:
        # Check file type
        if not resume.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        # Read uploaded file
        pdf_bytes = await resume.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        print(f"Received PDF: {resume.filename}, Size: {len(pdf_bytes)} bytes")

        # Extract text using PyMuPDF (fitz)
        try:
            pdf_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
            resume_text = "\n".join([page.get_text("text") for page in pdf_reader])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")

        print(f"Extracted Resume Text: {resume_text[:500]}...")  # Print first 500 chars for debugging

        # 🟢 Multiple API Calls for Different Aspects
        category_response = query_huggingface(
            "Categorize this resume into one of the following skills: Data Engineering, Data Science, Front-end, Java.",
            resume_text
        )
        
        key_areas_response = query_huggingface(
            "Identify key areas where the candidate has worked.",
            resume_text
        )

        contact_info_response = query_huggingface(
            "Extract contact details such as email and phone number from the resume.",
            resume_text
        )

        # 🟢 Combined Response
        return {
            "Category": category_response,
            "Key_Areas": key_areas_response,
            "Contact_Info": contact_info_response
        }

    except HTTPException as http_err:
        print(f"HTTP Error: {http_err.detail}")
        raise http_err

    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
            json={"inputs": f"Analyze this resume and help us to identify key areas where resource has worked : {resume_text}"}
        )

        return response.json()

    except HTTPException as http_err:
        print(f"HTTP Error: {http_err.detail}")
        raise http_err

    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
