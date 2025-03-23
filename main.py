from fastapi import FastAPI, File, UploadFile
from typing import List
import requests
import fitz  # PyMuPDF

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_KEY = "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"  # Store API Key as Environment Variable

@app.post("/rank_resumes/")
async def rank_resumes(files: List[UploadFile] = File(...)):
    ranked_resumes = []  # Store processed resumes

    for file in files:
        try:
            # Read PDF file
            pdf_bytes = await file.read()
            pdf_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
            resume_text = "\n".join([page.get_text("text") for page in pdf_reader])

            # Debug: Print extracted resume text
            print(f"DEBUG: Extracted text from {file.filename} - {resume_text[:300]}...")

            # Call Hugging Face API
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": f"Extract key areas from resume: {resume_text}"}
            )

            # Ensure response is a dictionary
            api_result = response.json()
            if isinstance(api_result, list):
                print(f"ERROR: API returned a list instead of a dictionary: {api_result}")
                continue  # Skip this resume

            # Construct ranked resume data
            ranked_resumes.append({
                "Filename": file.filename,
                "Category": api_result.get("Category", "Unknown"),
                "Experience_Years": api_result.get("Experience_Years", 0),
                "Key_Skills": api_result.get("Key_Skills", []),
                "Education": api_result.get("Education", "Unknown"),
                "Certifications": api_result.get("Certifications", []),
                "Rank_Score": api_result.get("Rank_Score", 0)
            })

        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")

    print("DEBUG: Final ranked resumes ->", ranked_resumes)  # Print ranked resumes before returning
    return {"ranked_resumes": ranked_resumes}
