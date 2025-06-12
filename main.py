from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from model import compute_ear  # Keep only EAR detection

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    temp_file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run only EAR-based analysis (no YOLO)
    drowsiness_score, attention_score = compute_ear(temp_file_path)

    # Decide user status
    if drowsiness_score is not None and drowsiness_score >= 0.7:
        status = "fatigue detected"
    elif drowsiness_score is not None:
        status = "attentive"
    else:
        status = "no face detected"

    # Delete temporary image
    os.remove(temp_file_path)

    return {
        "status": status,
        "drowsiness_score": drowsiness_score,
        "attention_score": attention_score
    }
