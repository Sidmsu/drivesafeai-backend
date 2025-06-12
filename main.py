from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from model import compute_ear
from yolo_detector import detect_objects

app = FastAPI()

# Allow frontend requests from any domain (CORS)
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

    # Run both YOLOv5 object detection and EAR-based analysis
    yolo_detections, confidences = detect_objects(temp_file_path)
    drowsiness_score, attention_score = compute_ear(temp_file_path)

    # Decision logic for alert status
    if "yawn" in yolo_detections or (drowsiness_score is not None and drowsiness_score >= 0.7):
        status = "fatigue detected"
    elif "phone" in yolo_detections:
        status = "distracted"
    elif drowsiness_score is not None:
        status = "attentive"
    else:
        status = "no face detected"

    # Cleanup
    os.remove(temp_file_path)

    return {
        "status": status,
        "detections": yolo_detections,
        "confidence": confidences,
        "drowsiness_score": drowsiness_score,
        "attention_score": attention_score
    }
