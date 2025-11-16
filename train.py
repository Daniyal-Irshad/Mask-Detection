from ultralytics import YOLO

# Load a YOLOv8 model (base)  
model = YOLO("yolov8n.pt")  # or a larger model

# Train on your exported dataset  
results = model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8
)

# After training, save best model
model.val()
model.save("mask_detection_yolo.pt")
