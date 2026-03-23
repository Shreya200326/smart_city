# 🚗 Accident Detection using YOLOv8

## 📌 Overview

This project detects road accidents from video footage using a deep learning classification model based on YOLOv8. It processes videos, extracts frames, trains a model to classify scenes as **Crash** or **Normal**, and performs real-time inference with stable predictions.

---

## 🎯 Features

* 🎥 Extract frames from videos automatically
* 🧠 Train a YOLOv8 classification model
* ⚡ Real-time accident detection on videos
* 📊 Sliding window prediction for smoother results
* 📍 Optional location detection using coordinates

---

## 🛠️ Tech Stack

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* NumPy
* Requests

---

## 📂 Project Structure

```
Major_Project/
│
├── README.md
├── requirements.txt
├── data.yaml
│
├── dataset_prep/
│   └── extract_frames.py
│
├── src/
│   ├── train.py
│   ├── inference.py
│   └── location.py
│
├── data/
│   └── images/
│       ├── train/
│       └── val/
│
├── runs/
│   └── classify/
│
└── outputs/
```

---

## ⚙️ How It Works

### 1. Frame Extraction

* Videos are split into frames
* Frames are categorized into:

  * `crash`
  * `normal`
* Data is automatically split into training and validation sets

---

### 2. Model Training

* Uses YOLOv8 classification model
* Trained on extracted frames
* Learns to classify scenes as accident or normal

---

### 3. Inference

* Processes video frame by frame
* Uses a sliding window to avoid flickering predictions
* Displays:

  * 🔴 ALERT (Crash detected)
  * 🟢 Normal

---

## 🚀 Setup & Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/accident-detection.git
cd accident-detection
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Prepare Dataset

Place dataset in:

```
data/raw/
├── Crash-1500/
├── Normal/
```

Run:

```
python dataset_prep/extract_frames.py
```

---

### Step 2: Train Model

```
python src/train.py
```

---

### Step 3: Run Inference

```
python src/inference.py
```

---

### Step 4 (Optional): Get Location from Coordinates

```
python src/location.py
```

---

## 💻 How to Run in PyCharm

1. Open PyCharm

2. Click **Open Project** and select the project folder

3. Set Python Interpreter

   * Go to **File > Settings > Project > Python Interpreter**
   * Select your virtual environment or create a new one

4. Install dependencies

   * Open terminal in PyCharm
   * Run:

     ```
     pip install -r requirements.txt
     ```

5. Configure Run File

   * Right click on any script (for example `train.py`)
   * Click **Run 'train'**

6. Run pipeline step by step:

   * Run `extract_frames.py`
   * Then `train.py`
   * Then `inference.py`

---

## 📊 Output

* Trained model saved in:

```
runs/classify/
```

* Inference output example:

```
ALERT (0.87)
Normal (0.12)
```

---

## ⚠️ Notes

* Dataset is not included due to size
* Model weights (`.pt`) are not uploaded
* Update file paths as per your system

---

## 🔮 Future Improvements

* Real-time webcam detection
* Emergency alert integration
* Web deployment using Flask or Streamlit
* Improved accuracy with larger dataset
