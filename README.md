# Real-Time Hand Sign Recognition

A real-time static hand sign classification system using MediaPipe and PyTorch, designed for low-latency gesture recognition from webcam input.

---

## 🎯 Final Project Scope

- Static gesture recognition for:
  - Digits (0–9)
  - Alphabet (A–Z)
- Landmark-based feature extraction using MediaPipe
- Neural network classifier (MLP-based architecture)
- Real-time webcam inference with live prediction overlay
- Modular and production-ready ML architecture
- Extensible design for future deployment via API

---

## 🏗 System Architecture

Webcam Frame  
→ MediaPipe Hand Detection  
→ 21 Hand Landmarks (x, y, z)  
→ Feature Vector (63 values)  
→ Neural Network Classifier  
→ Predicted Gesture (0–9 / A–Z)

---

## 🛠 Tech Stack

- Python 3.11+
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- scikit-learn

---

## 📂 Project Structure

hand-sign-recognition/
│
├── data/                  # Landmark datasets
│   ├── raw/               # Raw collected landmark CSV files
│   └── processed/         # Cleaned / normalized datasets
│
├── models/                # Saved trained model weights (.pt files)
│
├── ml/                    # Core machine learning modules
│   ├── detection/         # MediaPipe hand detection logic
│   ├── features/          # Landmark extraction & preprocessing
│   ├── training/          # Dataset loader and model training pipeline
│   ├── inference/         # Real-time prediction logic
│   └── utils/             # Helper utilities (metrics, config, etc.)
│
├── tests/                 # Unit and integration tests
│
├── requirements.txt       # Python dependencies
└── README.md

---

## 🚧 Development Roadmap

### Phase 1 — MVP (Digits 0–9)
- [ ] MediaPipe landmark extraction
- [ ] Landmark dataset collection
- [ ] Train MLP classifier for 10 classes
- [ ] Real-time webcam inference
- [ ] Accuracy evaluation

### Phase 2 — Full Alphabet Expansion
- [ ] Expand dataset to A–Z
- [ ] Retrain model for 36 classes
- [ ] Improve model generalization
- [ ] Confusion matrix analysis

### Phase 3 — Optimization & Production Readiness
- [ ] Latency benchmarking
- [ ] Model optimization
- [ ] Docker support
- [ ] Optional API deployment

---

## 📊 Expected Performance

- Target Accuracy: > 90%
- Real-Time Inference Latency: < 20ms per frame
- Robust performance under varying lighting conditions

---

## 🎓 Motivation

This project explores the design of a real-time computer vision system that bridges research-grade machine learning with practical deployment considerations. It focuses on clean system architecture, modular ML design, and low-latency inference.

---

## 📌 Status

Project initialized.  
Phase 1 (Digits 0–9) implementation in progress.