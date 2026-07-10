# Smart Multi-Hazard Detection System

> An AI-powered IoT-based real-time hazard detection system that combines Computer Vision, Machine Learning, and IoT sensors for intelligent detection of fire, smoke, and water leakage.

---

## Overview

The **Smart Multi-Hazard Detection System** is an intelligent safety monitoring solution that integrates **Computer Vision**, **Machine Learning**, and **IoT sensors** to detect multiple environmental hazards in real time.

Unlike traditional threshold-based alarm systems, this project combines:

- Deep Learning-based image recognition
- Sensor anomaly detection
- Machine Learning-based hazard classification
- Sensor-Vision decision fusion
- Real-time monitoring dashboard

This hybrid architecture significantly improves detection reliability while reducing false alarms.

---

# Features

- 🔥 Fire Detection using Deep Learning
- 💨 Smoke Detection using Deep Learning
- 💧 Water Leakage Detection using Deep Learning
- 🌡 Temperature Monitoring
- 💨 Gas Monitoring
- 💧 Humidity Monitoring
- 🌞 Ambient Light Monitoring
- 🌪 Pressure Monitoring
- 📊 Machine Learning-based Sensor Intelligence
- 🎯 Multi-modal Decision Fusion
- 🚨 Real-time Hazard Alerts
- 🔊 Audio Warning System
- 📈 Live Monitoring Dashboard
- 📜 Hazard Event History
- ⚡ ESP32 IoT Integration

---

# Project Architecture

```
                    Camera
                       │
               Capture Frames
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
 Fire CNN        Smoke CNN      Water Leak CNN
        │              │              │
        └──────────────┼──────────────┘
                       │
              Vision Predictions
                       │

ESP32 Sensors -----------------------------┐
                                           │
Temperature                                │
Humidity                                   │
Gas                                        │
Pressure                                   │
LDR                                        │
                                           ▼
                           Random Forest Classifier
                             (Sensor Intelligence)

                     Sensor + Vision Fusion Engine
                              │
                     Hazard Decision Manager
                              │
                     Flask Backend API
                              │
                Live Monitoring Dashboard
```

---

# Technologies Used

## Programming Languages

- Python
- C++ (ESP32 Arduino)

## Machine Learning

- Scikit-Learn
- Random Forest Classifier

## Deep Learning

- PyTorch
- Torchvision
- EfficientNetV2-S
- Transfer Learning

## Computer Vision

- OpenCV
- PIL

## Backend

- Flask
- REST APIs

## Frontend

- HTML
- CSS
- JavaScript

## IoT Hardware

- ESP32
- DHT22 Temperature & Humidity Sensor
- MQ Gas Sensor
- BMP280 Pressure Sensor
- LDR Sensor
- Flame Sensor
- Buzzer

---

# System Workflow

## Step 1 : Sensor Data Collection

ESP32 continuously collects environmental parameters including:

- Temperature
- Humidity
- Gas Concentration
- Atmospheric Pressure
- Ambient Light

The readings are transmitted to the Flask server through HTTP requests.

---

## Step 2 : Vision Processing

A camera continuously captures images.

Three independent deep learning models process every frame.

### Fire Detection Model

Classes:

- No Fire
- Controlled Fire
- Uncontrolled Fire

---

### Smoke Detection Model

Classes:

- Smoke
- No Smoke

---

### Water Leakage Detection Model

Classes:

- Leak
- No Leak

---

## Step 3 : Sensor Intelligence

Instead of relying on fixed thresholds, the system first learns the recent environmental behavior using a rolling baseline.

For every new reading, anomaly scores are calculated.

Feature engineering is then performed to compute patterns such as:

- Temperature Rise
- Gas Rise
- Humidity Drop
- Pressure Change
- Light Variation

These engineered features are passed to a Random Forest classifier.

The classifier predicts:

- Safe
- Fire Risk
- Smoke Risk
- Water Leakage Risk

along with confidence scores.

---

## Step 4 : Decision Fusion

The system combines

- Computer Vision confidence
- Sensor ML confidence

to generate the final hazard prediction.

This hybrid approach greatly reduces false positives.

Example:

```
Vision Fire Confidence = 96%

Sensor Fire Confidence = 90%

↓

Final Hazard Confidence = Very High
```

---

## Step 5 : Hazard Management

The hazard manager maintains:

- Active Hazards
- Hazard History
- Event IDs
- Time Stamps
- Alert Priorities

This prevents unstable alert flickering.

---

## Step 6 : Dashboard

The Flask dashboard displays:

- Live Sensor Readings
- Camera Predictions
- Hazard Status
- Active Alerts
- Historical Events
- System Status

The dashboard updates continuously.

---

# Deep Learning Models

The project contains three independent CNN models.

## Fire Detection

Model:

- EfficientNetV2-S

Classes:

- No Fire
- Controlled Fire
- Uncontrolled Fire

---

## Smoke Detection

Model:

- EfficientNetV2-S

Classes:

- Smoke
- No Smoke

---

## Water Leakage Detection

Model:

- EfficientNetV2-S

Classes:

- Leak
- No Leak

---

# Training Pipeline

Each model follows the same training process.

Dataset

↓

Data Augmentation

↓

Transfer Learning (EfficientNetV2-S)

↓

Cross Entropy Loss

↓

Backpropagation

↓

Validation

↓

Best Model Saved

---

# Data Augmentation

The models are trained using:

- Random Crop
- Horizontal Flip
- Rotation
- Color Jitter
- Translation
- Normalization
- Random Erasing

This improves model generalization.

---

# Machine Learning Pipeline

Sensor Data

↓

Rolling Baseline Calculation

↓

Feature Engineering

↓

Random Forest Classifier

↓

Risk Prediction

↓

Fusion with Vision

↓

Final Hazard Decision

---

# Folder Structure

```
Smart Multi-Hazard Detection System/

│
├── Fire Detection CV/
│   ├── dataset/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── best_model.pt
│
├── Smoke Detection CV/
│   ├── dataset/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── best_model.pt
│
├── Water Leakage CV/
│   ├── dataset/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── best_model.pt
│
├── Flask Backend/
│
├── Dashboard/
│
├── ESP32 Firmware/
│
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Smart-Multi-Hazard-Detection-System.git
```

Navigate to the project

```bash
cd Smart-Multi-Hazard-Detection-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Flask server

```bash
python app.py
```

Upload the ESP32 firmware using Arduino IDE.

Open the dashboard in your browser.

---

# Future Improvements

- Thermal Camera Integration
- Gas Type Classification
- Mobile Application
- Cloud Deployment
- MQTT Communication
- Edge AI Deployment on NVIDIA Jetson
- YOLO-based Multi-Hazard Localization
- Hazard Severity Prediction
- Predictive Maintenance
- SMS/Email Emergency Notifications

---

# Applications

- Smart Homes
- Industries
- Warehouses
- Data Centers
- Laboratories
- Hospitals
- Commercial Buildings
- Educational Institutions

---

# Advantages

- Multi-modal hazard detection
- Reduced false alarms
- Real-time monitoring
- Adaptive sensor intelligence
- AI-assisted decision making
- Modular architecture
- Easily scalable
- Low-cost deployment

---
