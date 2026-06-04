# 📧 Spam Mail Predictor

A Machine Learning-powered web application that classifies emails as **Spam** or **Not Spam (Ham)** using **Natural Language Processing (NLP)** and a **Logistic Regression** model. The application provides a simple web interface and a Flask-based REST API for real-time spam detection.

---

## 🚀 Features

* Real-time spam email classification
* Flask-based REST API
* NLP preprocessing and text cleaning
* TF-IDF feature extraction
* Logistic Regression classifier
* Model serialization using Joblib
* Interactive web interface using HTML, CSS, and JavaScript
* Cross-Origin Resource Sharing (CORS) support

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask
* Flask-CORS

### Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF Vectorizer
* Joblib

### Frontend

* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```text
Spam-Mail-Predictor/
│
├── app.py                  # Flask API
├── train_model.py          # Model training script
├── Spam Mail Prediction.ipynb
├── index.html              # Frontend UI
│
├── spam_model.pkl          # Trained Logistic Regression model
├── vectorizer.pkl          # TF-IDF Vectorizer
│
├── Spam Mail Examples.txt
│
└── README.md
```

---

## ⚙️ Working Flow

1. User enters email content through the web interface.
2. Frontend sends the email text to the Flask API.
3. API preprocesses the text and converts it into TF-IDF features.
4. The trained Logistic Regression model performs inference.
5. The prediction result is returned as:

   * Spam
   * Not Spam

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing

* Lowercasing text
* Removing special characters
* Tokenization
* Text cleaning

### Feature Extraction

* TF-IDF Vectorization

### Classification Model

* Logistic Regression

### Performance

* Achieved **97% classification accuracy** on the dataset.

---

## 🔌 API Endpoint

### Predict Spam

**POST** `/predict`

#### Request

```json
{
    "email": "Congratulations! You have won a free iPhone."
}
```

#### Response

```json
{
    "result": "Spam"
}
```

---

## ▶️ Installation & Setup

### Clone Repository

```bash
git clone https://github.com/arjun250/Spam-Mail-Predictor.git
cd Spam-Mail-Predictor
```

### Install Dependencies

```bash
pip install flask
pip install flask-cors
pip install scikit-learn
pip install numpy
pip install joblib
```

### Run Application

```bash
python app.py
```

Server starts at:

```text
http://127.0.0.1:5000
```

---

## 📸 Screenshots

### Prediction Result - Spam Email

![Prediction Result - Spam Email](Screenshot%20(333).png)

### Prediction Result - Not Spam Email

![Prediction Result - Not Spam Email](Screenshot%20(334).png)

---

## ⭐ If you found this project useful, consider giving it a star.
