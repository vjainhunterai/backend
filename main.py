from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import fitz  # PyMuPDF for PDF text extraction

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_KEY = "your_huggingface_api_key_here"

def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file using PyMuPDF."""
    pdf_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text("text") for page in pdf_reader])

@app.post("/rank_resumes/")
async def rank_resumes(resumes: list[UploadFile] = File(...)):
    """Rank multiple resumes based on experience, skills, and certifications."""

    if not resumes:
        raise HTTPException(status_code=400, detail="No resumes uploaded.")

    ranked_candidates = []

    for resume in resumes:
        try:
            # Read and extract text from PDF
            pdf_bytes = await resume.read()
            resume_text = extract_text_from_pdf(pdf_bytes)

            if not resume_text.strip():
                continue  # Skip if no text extracted

            # Call Hugging Face API for structured analysis
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": f"Analyze and categorize this resume:\n{resume_text}"}
            )

            analysis = response.json()

            # Example data extraction (modify based on actual API response)
            candidate_data = {
                "Filename": resume.filename,
                "Category": analysis.get("Category", "Unknown"),
                "Experience_Years": analysis.get("Experience_Years", 0),
                "Key_Skills": analysis.get("Key_Skills", []),
                "Education": analysis.get("Education", "Not mentioned"),
                "Certifications": analysis.get("Certifications", []),
                "Rank_Score": (
                    (analysis.get("Experience_Years", 0) * 0.5) +
                    (len(analysis.get("Key_Skills", [])) * 0.3) +
                    (len(analysis.get("Certifications", [])) * 0.2)
                )
            }

            ranked_candidates.append(candidate_data)

        except Exception as e:
            print(f"Error processing {resume.filename}: {e}")

    # Rank candidates based on the calculated Rank_Score
    ranked_candidates = sorted(ranked_candidates, key=lambda x: x["Rank_Score"], reverse=True)

    return {"ranked_resumes": ranked_candidates}
