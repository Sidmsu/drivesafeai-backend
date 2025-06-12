from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from model import compute_ear
from yolo_detector import detect_objects

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    temp_file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run both YOLO + EAR
    yolo_detections, confidences = detect_objects(temp_file_path)
    drowsiness_score, attention_score = compute_ear(temp_file_path)

    # Decide status based on both
    if "yawn" in yolo_detections or (drowsiness_score is not None and drowsiness_score >= 0.7):
        status = "fatigue detected"
    elif "phone" in yolo_detections:
        status = "distracted"
    elif drowsiness_score is not None:
        status = "attentive"
    else:
        status = "no face detected"

    os.remove(temp_file_path)

    return {
        "status": status,
        "detections": yolo_detections,
        "confidence": confidences,
        "drowsiness_score": drowsiness_score,
        "attention_score": attention_score
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Fallback to 8000 for local
    uvicorn.run(app, host="0.0.0.0", port=port)
