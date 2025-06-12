import torch

# Load once
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def detect_objects(image_path):
    results = model(image_path)
    detections = results.pandas().xyxy[0]
    labels = list(detections['name'])
    confidence = list(detections['confidence'])
    return labels, confidence
