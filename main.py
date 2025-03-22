from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import fitz  # PyMuPDF for PDF text extraction
import os
import re

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

def extract_experience_years(response_text):
    """Extract numeric experience years from response"""
    match = re.search(r"(\d+(\.\d+)?)\s*(years|yrs|year)", response_text, re.IGNORECASE)
    return float(match.group(1)) if match else 0  # Default to 0 if not found

def calculate_rank_score(experience, skills_count, cert_count):
    """Calculate weighted rank score"""
    return (experience * 0.5) + (skills_count * 0.3) + (cert_count * 0.2)

@app.post("/rank_resumes/")
async def rank_resumes(resumes: list[UploadFile] = File(...)):
    """Upload multiple resumes and rank them based on experience, skills, and certifications"""
    try:
        ranked_candidates = []

        for resume in resumes:
            # Check file type
            if not resume.filename.endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{resume.filename}: Only PDF files are allowed.")

            # Read uploaded file
            pdf_bytes = await resume.read()
            if not pdf_bytes:
                raise HTTPException(status_code=400, detail=f"{resume.filename}: Uploaded file is empty.")

            print(f"Received PDF: {resume.filename}, Size: {len(pdf_bytes)} bytes")

            # Extract text using PyMuPDF (fitz)
            try:
                pdf_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
                resume_text = "\n".join([page.get_text("text") for page in pdf_reader])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{resume.filename}: Invalid PDF file: {str(e)}")

            if not resume_text.strip():
                raise HTTPException(status_code=400, detail=f"{resume.filename}: No text could be extracted.")

            print(f"Extracted Resume Text: {resume_text[:500]}...")  # Debugging

            # 🔹 Analyze Resume
            category_response = query_huggingface(
                "Categorize this resume into one of the following: Data Engineering, Data Science, Front-end, Java.",
                resume_text
            )

            experience_response = query_huggingface(
                "Extract total years of experience from this resume.",
                resume_text
            )

            key_areas_response = query_huggingface(
                "Identify key technical skills mentioned in this resume.",
                resume_text
            )

            education_response = query_huggingface(
                "Extract highest education qualification from this resume.",
                resume_text
            )

            certifications_response = query_huggingface(
                "Extract certifications from this resume.",
                resume_text
            )

            # Convert experience to a numeric format
            experience_years = extract_experience_years(experience_response.get("experience", ""))

            # Count key skills and certifications (assuming they return a list)
            skills_count = len(key_areas_response.get("skills", []))
            cert_count = len(certifications_response.get("certifications", []))

            # Calculate weighted score
            rank_score = calculate_rank_score(experience_years, skills_count, cert_count)

            # Store results
            ranked_candidates.append({
                "Filename": resume.filename,
                "Category": category_response.get("category", "Unknown"),
                "Experience_Years": experience_years,
                "Key_Skills": key_areas_response.get("skills", []),
                "Education": education_response.get("education", "Unknown"),
                "Certifications": certifications_response.get("certifications", []),
                "Rank_Score": rank_score
            })

        # 🔹 Rank Resumes by Weighted Score
        ranked_candidates = sorted(ranked_candidates, key=lambda x: x["Rank_Score"], reverse=True)

        return {"Ranked_Resumes": ranked_candidates}

    except HTTPException as http_err:
        print(f"HTTP Error: {http_err.detail}")
        raise http_err

    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
