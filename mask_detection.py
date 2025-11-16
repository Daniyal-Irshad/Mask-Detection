import cv2
from ultralytics import YOLO
import os

# Load your trained model
model = YOLO("yolov8n.pt")  # or yolov8n.pt if using pretrained
# model = YOLO("mask_detection_yolo.pt")  # or yolov8n.pt if using pretrained


# Create output folder if not exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ----- IMAGE INFERENCE -----
def run_on_image(image_path):
    results = model.predict(image_path, imgsz=640, device="cpu")  # force CPU
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    results[0].save(save_path)
    print(f"Saved result to {save_path}")

# ----- VIDEO / WEBCAM INFERENCE -----
def run_on_video(source=0):  # 0 for webcam, or path to video file
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback to 30 if 0

    # Output video writer
    output_path = os.path.join(output_dir, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Press 'q' to quit, 'p' to pause frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model.predict(frame, imgsz=640, device="cpu")[0]

        # Get annotated frame
        annotated_frame = results.plot()

        # Show live
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Save frame to video
        out.write(annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            cv2.waitKey(0)  # pause until any key

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved processed video to {output_path}")

# --------- USAGE EXAMPLES ---------
# Images
# run_on_image("test1.jpg")
# run_on_image("test2.jpg")

# Video file
# run_on_video("test_video.mp4")

# Webcam
run_on_video(0)
