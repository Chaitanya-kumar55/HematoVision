# 🩸 HematoVision: Advanced Blood Cell Classification Using Transfer Learning

HematoVision is a deep learning-powered web application designed for **automated blood cell classification**. Leveraging the power of **transfer learning** with pre-trained convolutional neural networks (CNNs), HematoVision identifies and classifies various blood cell types with high accuracy, supporting diagnostic applications in medical labs, telemedicine, and educational institutions.

---

## 📌 Table of Contents

- [🚀 Project Overview](#-project-overview)
- [🎯 Purpose](#-purpose)
- [🧠 Problem Statement](#-problem-statement)
- [🛠️ Features](#-features)
- [🖼️ Sample Workflow](#-sample-workflow)
- [📊 Technologies Used](#-technologies-used)
- [🧪 Testing and Evaluation](#-testing-and-evaluation)
- [💡 Results](#-results)
- [✅ Advantages](#-advantages)
- [⚠️ Limitations](#️-limitations)
- [🔭 Future Scope](#-future-scope)
- [📁 Project Structure](#-project-structure)
- [💻 How to Run Locally](#-how-to-run-locally)
- [🔗 Demo & GitHub](#-demo--github)
- [📂 Dataset](#-dataset)

---

## 🚀 Project Overview

HematoVision classifies four major types of blood cells:
- **Eosinophils**
- **Lymphocytes**
- **Monocytes**
- **Neutrophils**

The system uses a pre-trained CNN model (e.g., **ResNet50** or **VGG16**) via **Transfer Learning** to provide accurate, scalable, and rapid diagnostics through a simple web interface built with **Flask**.

---

## 🎯 Purpose

To deliver a reliable, scalable, and efficient **AI-based blood cell classification** tool to support:
- Clinical diagnostics
- Telemedicine workflows
- Medical training and education

---

## 🧠 Problem Statement

Manual classification of blood cells is:
- Time-consuming
- Prone to human error
- Dependent on expert availability

HematoVision automates this with AI, delivering consistent and fast results.

---

## 🛠️ Features

- 🔍 Upload blood cell images for instant classification
- ⚙️ Backend powered by Flask + Keras/TensorFlow
- 📈 Accuracy-optimized through Transfer Learning
- 💬 Confidence scores with predictions
- 🌐 Web-friendly frontend (HTML/CSS/JS)

---

## 🖼️ Sample Workflow

![Workflow](static/Screenshot%202025-06-30%20181026.png)


---

## 📊 Technologies Used

- **Language:** Python
- **Frameworks:** Flask, TensorFlow/Keras
- **Frontend:** HTML, CSS, JavaScript
- **Model:** ResNet50 (or VGG16 / MobileNetV2)
- **Dataset:** 12,000 labeled blood cell images

---

## 🧪 Testing and Evaluation

- ✔️ Accuracy tested on validation/test sets
- ⏱️ Prediction time evaluated under different image resolutions
- 🧪 Functional and performance testing included

---

## 💡 Results

- 🧠 Achieved **high accuracy** on test set
- 📉 Reduced classification time significantly
- 📸 User-friendly interface for non-technical users

**(Screenshots can be found in `/screenshots`)**

---

## ✅ Advantages

- ✅ High accuracy via pre-trained CNN
- ✅ Reduces manual workload and delays
- ✅ Easily integrable into clinical pipelines

---

## ⚠️ Limitations

- ❌ Requires a GPU for model training
- ❌ Relies on annotated, high-quality image data
- ❌ Doesn't use biochemical markers, only image-based features

---

## 🔭 Future Scope

- 🔬 Support for abnormal/malignant cells (e.g., leukemia)
- ☁️ Cloud and mobile deployment
- 🏥 Integration with hospital systems
- 🔴 Real-time image classification from microscopy cameras

---

## 📁 Project Structure

HematoVision/
│
├── app.py # Main Flask app
├── hemato_model.h5 # Trained model file
├── static/ # CSS, JS files
├── templates/ # HTML templates
├── preprocessing.py # Image preprocessing utilities
├── requirements.txt # Python dependencies
├── screenshots/ # Output screenshots
└── README.md # Project documentation


---

## 💻 How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/hematovision.git
cd hematovision
