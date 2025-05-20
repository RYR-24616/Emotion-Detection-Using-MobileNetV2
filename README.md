Emotion-Detection-Using-MobileNetV2
A deep learning-based image classification model that detects human emotions (like happy, sad, angry, etc.) from facial images. Leveraged transfer learning using MobileNetV2 and trained it on a labeled emotion dataset with TensorFlow and Keras. The model achieved reliable accuracy and supports real-time inference for emotion classification.

This is a deep learning-based image classification project that detects human emotions like happy, sad, angry, neutral, surprise, etc., from facial images. It started as a simple experimental project and evolved using transfer learning with the MobileNetV2 architecture—known for being fast, lightweight, and accurate.

The model is trained using TensorFlow and Keras on a labeled dataset of facial emotions. It takes in an image and classifies it into one of the predefined emotion categories. You can also use the trained model for real-time inference with your own images.

📂 Dataset
The dataset used for training and testing the model is a labeled collection of facial emotion images.

📥 Click here to download the dataset from Google Drive
https://drive.google.com/drive/folders/1lLbJjyaTqV__oiA-Y-ddYSCgGVJoRM0k?usp=drive_link

⚠️ Limitations
While the model performs well overall, emotions like "fear" and "surprise" can sometimes be misclassified. This is because facial expressions alone don’t always capture the full context—body language and posture also play a key role in distinguishing subtle emotions. So expect some limitations in those categories.
