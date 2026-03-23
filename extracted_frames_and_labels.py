import os
import cv2
import random
from pathlib import Path

ROOT = r"C:\datsets"
CRASH_DIR = os.path.join(ROOT, "Crash-1500")
NORMAL_DIR = os.path.join(ROOT, "Normal")

OUT_DIR = os.path.join(Path(__file__).parent, "data")


VAL_SPLIT = 0.10
MAX_FRAMES_PER_VIDEO = 50
SAMPLE_EVERY_N = 1

IMG_TRAIN_CRASH = os.path.join(OUT_DIR, "images", "train", "crash")
IMG_TRAIN_NORMAL = os.path.join(OUT_DIR, "images", "train", "normal")
IMG_VAL_CRASH = os.path.join(OUT_DIR, "images", "val", "crash")
IMG_VAL_NORMAL = os.path.join(OUT_DIR, "images", "val", "normal")

# Create all the necessary output directories.
for p in [IMG_TRAIN_CRASH, IMG_TRAIN_NORMAL, IMG_VAL_CRASH, IMG_VAL_NORMAL]:
    os.makedirs(p, exist_ok=True)

def split_list(lst, val_ratio):
    """Shuffles and splits a list into training and validation sets."""
    random.shuffle(lst)
    # Ensure at least one video is in the validation set.
    nval = max(1, int(len(lst) * val_ratio))
    return lst[nval:], lst[:nval]

def process_video(video_path, out_img_folder):
    """
    Extracts frames from a single video and saves them as JPG images.

    Args:
        video_path (str): The full path to the input video file.
        out_img_folder (str): The directory where the extracted frames will be saved.

    Returns:
        int: The number of frames successfully saved from the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0

    saved_count = 0
    frame_idx = 0
    base_name = Path(video_path).stem  # Gets the filename without the extension (e.g., "000001")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Apply sampling logic
        if frame_idx % SAMPLE_EVERY_N == 0:
            # Stop if the maximum number of frames has been reached for this video
            if MAX_FRAMES_PER_VIDEO and saved_count >= MAX_FRAMES_PER_VIDEO:
                break

            # Define the output image name and full path
            img_name = f"{base_name}_{frame_idx:03d}.jpg"
            img_out_path = os.path.join(out_img_folder, img_name)

            # Save the frame as a JPG image
            cv2.imwrite(img_out_path, frame)
            saved_count += 1
            print(f"Saved frame {frame_idx} from {Path(video_path).name} -> {img_out_path}")

        frame_idx += 1

    cap.release()
    return saved_count

# -------------------------------------------------

# 1. Get lists of all video files
print("Scanning video directories...")
crash_videos = sorted([os.path.join(CRASH_DIR, f) for f in os.listdir(CRASH_DIR) if f.endswith(".mp4")])
normal_videos = sorted([os.path.join(NORMAL_DIR, f) for f in os.listdir(NORMAL_DIR) if f.endswith(".mp4")])

# 2. Split videos into training and validation sets
crash_train, crash_val = split_list(crash_videos, VAL_SPLIT)
normal_train, normal_val = split_list(normal_videos, VAL_SPLIT)

print("-" * 30)
print(f"Crash videos -> Train: {len(crash_train)}, Validation: {len(crash_val)}")
print(f"Normal videos -> Train: {len(normal_train)}, Validation: {len(normal_val)}")
print("-" * 30)

# 3. Process all videos and save the frames
print("\nProcessing CRASH videos for TRAINING set...")
for v in crash_train:
    process_video(v, IMG_TRAIN_CRASH)

print("\nProcessing CRASH videos for VALIDATION set...")
for v in crash_val:
    process_video(v, IMG_VAL_CRASH)

print("\nProcessing NORMAL videos for TRAINING set...")
for v in normal_train:
    process_video(v, IMG_TRAIN_NORMAL)

print("\nProcessing NORMAL videos for VALIDATION set...")
for v in normal_val:
    process_video(v, IMG_VAL_NORMAL)

# 4. Print final counts of the generated image files
print("-" * 30)
print("\nFinal counts of extracted frames:")
for folder in [IMG_TRAIN_CRASH, IMG_TRAIN_NORMAL, IMG_VAL_CRASH, IMG_VAL_NORMAL]:
    count = len(os.listdir(folder))
    print(f"- {folder}: {count} images")

print(f"\n✅ Done! All frames are saved in: {OUT_DIR}")
