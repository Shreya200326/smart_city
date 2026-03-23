from ultralytics import YOLO

# Directory that contains train/ and val/ subfolders
DATA_DIR = "data/images"
EPOCHS = 2
BATCH = 32
IMG_SIZE = 224

# Use a classification model (yolov8n-cls.pt is designed for classification)
model = YOLO("yolov8n-cls.pt")

print("Starting classification training...")
model.train(
    data=DATA_DIR,\
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMG_SIZE,
    task="classify",
    name="ccd-classify"
)
print("Training finished.")
