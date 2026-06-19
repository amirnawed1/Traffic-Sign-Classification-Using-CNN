# 🚦 Traffic Sign Classification Using CNN

A Deep Learning project for classifying traffic signs using Convolutional Neural Networks (CNN) and Transfer Learning (MobileNetV2).

The project was developed using TensorFlow and deployed using Streamlit.

---

## 📌 Project Overview

Traffic sign recognition plays an important role in intelligent transportation systems and autonomous vehicles.

This project aims to classify traffic signs into 43 classes using deep learning techniques.

Two models were implemented and compared:

- Custom CNN
- MobileNetV2 (Transfer Learning)

The CNN model achieved the best performance.

---

## 📊 Model Performance

| Model | Validation Accuracy |
|---------|---------|
| CNN | 99.39% |
| MobileNetV2 | 39.91% |

### Final Selected Model

✅ CNN

---

## 🗂 Dataset

Dataset Used:

German Traffic Sign Recognition Benchmark (GTSRB)

- 39,209 training images
- 12,630 testing images
- 43 traffic sign classes

---

## 🔍 Exploratory Data Analysis

The project includes:

- Class distribution analysis
- Sample image visualization
- Image size distribution
- RGB color analysis
- Pixel intensity distribution

---

## ⚙️ Data Preprocessing

- Image resizing
- Pixel normalization
- One-hot encoding
- Train-validation split

---

## 🧠 Deep Learning Models

### 1. Custom CNN

- Conv2D
- MaxPooling2D
- Flatten
- Dense Layer
- Dropout

### 2. MobileNetV2

Transfer Learning with pretrained ImageNet weights.

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-Learn
- Streamlit

---

## 🚀 Streamlit Deployment

The trained CNN model was deployed using Streamlit.

Features:

- Upload traffic sign image
- Predict class
- Confidence score
- Interactive dashboard

---

## 📂 Project Structure

```

Traffic-Sign-Classification-Using-CNN
│
├── app.py
├── best_model.keras
├── requirements.txt
├── README.md
├── Traffic_Sign_Classification_Project.ipynb
└── LICENSE

```

---

## 🔮 Future Improvements

- Data augmentation
- EfficientNet
- ResNet
- Real-time traffic sign recognition
- Streamlit dashboard enhancements

---

## ❤️ Author

Developed using TensorFlow and Streamlit.
