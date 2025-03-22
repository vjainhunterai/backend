from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import fitz  # PyMuPDF for PDF text extraction

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_KEY = "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"  # Store API Key as Environment Variable

@app.post("/rank_resumes/")
async def rank_resumes(resumes: list[UploadFile] = File(...)):
    """Rank multiple resumes based on experience, skills, and certifications"""

    if not resumes:
        raise HTTPException(status_code=400, detail="No resumes uploaded.")

    ranked_candidates = []

    for resume in resumes:
        try:
            # Read and extract text from PDF
            pdf_bytes = await resume.read()
            pdf_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
            resume_text = "\n".join([page.get_text("text") for page in pdf_reader])

            if not resume_text.strip():
                continue  # Skip if no text extracted

            # Call Hugging Face API for ranking criteria
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": f"Analyze and rank this resume: {resume_text}"}
            )

            analysis = response.json()
            ranked_candidates.append({"name": resume.filename, "analysis": analysis})

        except Exception as e:
            print(f"Error processing {resume.filename}: {e}")

    # Apply ranking logic (dummy sorting by length of text)
    ranked_candidates = sorted(ranked_candidates, key=lambda x: len(x["analysis"]), reverse=True)

    return {"ranked_resumes": ranked_candidates}
