import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque


MODEL_PATH = "runs/classify/ccd-classify4/weights/best.pt"


VIDEO_PATH = r"C:\datsets\Crash-1500\000127.mp4"


WINDOW_SECONDS = 2.0  # Seconds of video to average for a stable prediction.
FPS = 10              # The FPS of the original dataset videos.
CONFIDENCE_THRESHOLD = 0.50 # The confidence score needed to label a segment as a crash.
# ---------------------

# 1. Load the trained YOLOv8 model
print(f"Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file not found at '{MODEL_PATH}'. Did you run training?")
model = YOLO(MODEL_PATH)

# You can uncomment the line below to verify the class order the model learned.
# It should print: {0: 'crash', 1: 'normal'}
# print("Model class names:", model.names)

# 2. Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Error: Cannot open video file '{VIDEO_PATH}'")

# 3. Set up the sliding window
# This helps to smooth out predictions and avoid "flickering" between labels.
video_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
win_frames = max(1, int(WINDOW_SECONDS * video_fps))
scores = deque(maxlen=win_frames)

print("Starting video inference... Press 'q' to quit.")
while True:
    # Read one frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # End of the video

    # 4. Get model prediction for the current frame
    # The model expects a square image, but YOLOv8 handles the resizing automatically.
    results = model.predict(source=frame, verbose=False, imgsz=224)

    # 5. Extract the crash probability
    crash_prob = 0.0
    try:
        # CORRECTED CODE:
        # Access the underlying probability tensor via the .data attribute,
        # move it to the CPU, and then convert to a NumPy array.
        probs_tensor = results[0].probs.data
        probs_numpy = probs_tensor.cpu().numpy()

        # The model learns classes in alphabetical order. 'crash' comes before 'normal',
        # so the output probabilities are [crash_prob, normal_prob].
        # We need the probability at index 0 for the 'crash' class.
        crash_prob = float(probs_numpy[0])
    except (AttributeError, IndexError) as e:
        print(f"Warning: Could not extract probability from model output. Details: {e}")

    # Add the current frame's score to our sliding window
    scores.append(crash_prob)

    # Calculate the average probability over the window
    avg_prob = np.mean(scores)

    # 6. Determine the label and color for the display text
    label = "ALERT" if avg_prob >= CONFIDENCE_THRESHOLD else "Normal"
    color = (0, 0, 255) if label == "ALERT" else (0, 255, 0) # Red for crash, Green for normal

    # 7. Draw the text on the frame
    display_text = f"{label} ({avg_prob:.2f})"
    cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # 8. Display the frame in a window
    cv2.imshow("Crash Detection Inference", frame)

    # Allow the user to quit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
print("Inference finished.")

