# 🧠 FocusAI: AI-Based Student Attention Tracking System

FocusAI is an intelligent, real-time student attention tracking system designed to monitor, analyze, and enhance student engagement during learning sessions. It leverages advanced Computer Vision techniques (OpenCV and MediaPipe) to detect facial orientation and eye movements, classifying a student's attention level automatically.

## 🚀 Problem Statement

Maintaining consistent student attention during learning sessions has become a significant challenge, especially with remote learning systems. Traditional methods of monitoring student engagement are subjective and not scalable. FocusAI automates this by providing real-time, non-intrusive tracking using AI.

## ✨ Features

- **Real-Time Attention Tracking:** Monitors students continuously via webcam.
- **Head Pose Estimation:** Detects if the student is looking away from the screen.
- **Eye Tracking (EAR):** Calculates Eye Aspect Ratio to detect drowsiness or closed eyes.
- **Attention Logic System:** Classifies state into:
  - ✅ **Focused** (Looking at screen, eyes open)
  - ❌ **Distracted** (Looking away)
  - 😴 **Drowsy** (Eyes closed)
- **Interactive Dashboard:** Beautiful Streamlit-based UI displaying live video, attention metrics, score %, and history logs.

## 🛠️ Tech Stack

- **Python 3.8+**
- **OpenCV** (Video capture & image processing)
- **MediaPipe** (Face Mesh & 468 3D landmarks)
- **NumPy & Pandas** (Data handling)
- **Streamlit** (Web Dashboard UI)

## 📦 Project Structure

```
attention-tracking-system/
│
├── utils/
│   ├── face_detection.py      # MediaPipe Face Mesh initialization and drawing
│   ├── eye_tracking.py        # EAR calculation and Head Pose estimation
│   ├── attention_logic.py     # Classification rules (Focused/Distracted/Drowsy)
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## ⚙️ How to Run

1. **Clone the repository or download the source code.**
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
4. Ensure your webcam is connected and grant browser permissions if requested.

## 🏆 Presentation Highlights

- **Problem:** Students lose focus, which affects academic performance.
- **Solution:** FocusAI provides an automated, non-invasive attention tracker.
- **Impact:** Helps students and educators identify patterns of distraction and improve overall productivity.
