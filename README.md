# 😊 Emotion Detection Using MobileNetV2

A deep learning-based image classification model that detects human emotions — such as **happy**, **sad**, **angry**, **neutral**, and **surprise** — from facial images in real time. Built using **TensorFlow**, **Keras**, and **transfer learning with MobileNetV2**, the model is lightweight, fast, and accurate for practical applications.

---

## 📌 Project Description

This project began as a simple experiment and evolved into a powerful real-time emotion recognition system.

We used **MobileNetV2**, a pre-trained convolutional neural network, for its balance of speed and performance. The model was fine-tuned on a labeled dataset of facial emotions, allowing it to classify expressions into predefined categories with high reliability.

You can use the trained model to:
- Detect emotions from still images.
- Perform real-time emotion detection using a webcam.

---

## 🧠 How It Works

- Input: Facial image
- Output: Predicted emotion label (e.g., "Happy", "Sad", "Angry", etc.)
- Backend: MobileNetV2 + Custom classifier head
- Frameworks: TensorFlow + Keras
- Deployment: Can be integrated into real-time applications using OpenCV

---

## 📁 Dataset

The dataset used is a **labeled collection of facial images**, each annotated with the corresponding emotion.

📥 **Download the dataset**:  
[Google Drive Link](https://drive.google.com/drive/folders/1lLbJjyaTqV__oiA-Y-ddYSCgGVJoRM0k?usp=drive_link)

---

## ✅ Features

- 🔁 Transfer Learning using **MobileNetV2**
- 🧠 Trained on real-world facial emotion images
- 📸 Real-time detection with webcam support
- 🖼️ Visual output of predictions with emotion labels

---

## ⚠️ Limitations

While the model performs well overall, there are some known limitations:

- Emotions like **"fear"** and **"surprise"** can be occasionally misclassified.
- Facial expressions alone may not capture subtle or overlapping emotions, as **context, body language, and tone** also play critical roles in emotional perception.

Use this tool as a **baseline emotion indicator**, not a definitive diagnosis.

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **OpenCV**
- **MobileNetV2**
- **NumPy**, **Matplotlib**

---
PS:I have used Chatgpt to Create the Read me file to make it more Readable
---

