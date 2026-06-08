# 🦴 Bone Fracture Detection Using CNN

## 📌 Overview

Bone Fracture Detection Using CNN is a deep learning-based web application that automatically detects bone fractures from X-ray images. The system utilizes Convolutional Neural Networks (CNNs) to classify X-ray images as **Fractured** or **Normal**, providing fast and reliable diagnostic assistance.

The project integrates a trained CNN model with a Flask web application, allowing users to upload X-ray images and receive real-time predictions along with confidence scores.

---

## 🚀 Features

* Upload X-ray images through a user-friendly web interface
* Automated bone fracture detection using CNN
* Real-time prediction with confidence score
* Image preprocessing and normalization
* Responsive and modern UI
* Flask-based deployment

---

## 🛠️ Technologies Used

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Python
* Flask

### Deep Learning

* TensorFlow
* Keras
* CNN (Convolutional Neural Network)

### Libraries

* NumPy
* OpenCV
* Matplotlib
* Pillow

---

## 📂 Project Structure

```text
Bone-Fracture-Detection/
│
├── app.py
├── model/
│   ├── cnn_model.py
│   ├── predict.py
│   └── saved_model/
│       └── bone_fracture_cnn.h5
│
├── dataset/
│   ├── train/
│   └── test/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── uploads/
│
├── train_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Bone-Fracture-Detection.git
cd Bone-Fracture-Detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
python train_model.py
```

The trained model will be saved as:

```text
model/saved_model/bone_fracture_cnn.h5
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 📊 Workflow

1. User uploads an X-ray image.
2. Image is preprocessed and resized.
3. CNN model extracts image features.
4. Model predicts fracture status.
5. Confidence score is generated.
6. Result is displayed on the web interface.

---

## 🎯 Project Objectives

* Automate bone fracture detection from X-ray images.
* Reduce diagnosis time and human error.
* Assist healthcare professionals with AI-based predictions.
* Improve accessibility to fracture screening tools.
* Provide a simple and interactive diagnostic interface.

---

## 🔮 Future Enhancements

* Multi-class fracture classification.
* Fracture localization using Grad-CAM.
* Integration with hospital management systems.
* Mobile application deployment.
* Cloud-based prediction services.

---

## 👨‍💻 Team Members

* D. Vasantha
* B. Varshitha
* J. Anand
* B. Chandrika

---

## 📜 License

This project is developed for academic and educational purposes.
