# MediChat: Medical Symptom Classification Chatbot

## Project Overview

MediChat is a medical symptom classification chatbot built with Python. The application takes user-entered symptoms, processes the text using NLP techniques, and predicts a possible medical condition using a trained PyTorch classification model. The chatbot also provides basic health advice based on the predicted condition.

This project was developed as an academic/group project and is being refined into a portfolio-ready machine learning application.

> **Medical Disclaimer:**  
> This application is for educational and portfolio purposes only. It is not a medical diagnosis tool and should not be used as a replacement for professional medical advice. Users should consult a qualified healthcare professional for any medical concerns.

---

## Key Features

- Symptom-based disease prediction
- Text preprocessing using NLTK
- TF-IDF vectorization for symptom text
- PyTorch-based classification model
- Gradio web interface
- Basic input cleaning and invalid input handling
- Disease-specific advice after prediction
- Supports multiple disease classes

---

## Tech Stack

- Python 3.10
- Pandas
- Scikit-learn
- NLTK
- PyTorch
- Gradio

---

## Project Structure

```text
medical-symptom-chatbot/
│
├── App.py                 # Main Gradio application
├── model.py               # PyTorch model architecture
├── utils.py               # Text preprocessing and vectorizer utilities
├── preprocess_data.py     # Dataset and DataLoader helper functions
├── Symptoms.csv           # Symptom dataset
├── saved_model.pth        # Trained PyTorch model weights
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
└── screenshots/           # Optional app screenshots
```

---

## How It Works

1. The user enters symptoms into the chatbot interface.
2. The input text is cleaned and preprocessed.
3. Symptoms are collected until enough information is available.
4. The combined symptoms are transformed using TF-IDF vectorization.
5. The trained PyTorch model predicts the most likely disease class.
6. The chatbot returns the predicted condition with basic advice.

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ahmadnawaz27/medical-symptom-chatbot.git
cd medical-symptom-chatbot
```

### 2. Create a virtual environment

```bash
py -3.10 -m venv .venv
```

### 3. Activate the environment

For Windows PowerShell:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Download required NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 6. Run the application

```bash
python App.py
```

Open the local Gradio URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

---

## Example Usage

Example symptom inputs:

```text
fever cough headache
```

```text
skin rash itching redness
```

```text
joint pain swelling stiffness
```

The chatbot will process the symptoms and return a possible condition with general advice.

---

## Current Limitations

- The chatbot provides possible predictions only, not confirmed diagnoses.
- The model depends on the quality and coverage of the training dataset.
- It may struggle with vague, incomplete, or uncommon symptom descriptions.
- The current model uses TF-IDF features rather than a full conversational medical reasoning system.
- Confidence thresholding and model evaluation reporting should be improved in future versions.

---

## Future Improvements

- Save and load the fitted TF-IDF vectorizer instead of refitting during runtime
- Add prediction confidence scores
- Add a minimum confidence threshold before returning a condition
- Improve symptom extraction and input validation
- Add model evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix
- Improve UI design and add screenshots
- Add deployment instructions
- Add safer medical response wording

---

## Portfolio Value

This project demonstrates:

- Natural Language Processing fundamentals
- Text classification workflow
- PyTorch model integration
- Machine learning application deployment using Gradio
- Modular Python project structure
- Practical user-facing ML interface development

---

## Author

Ahmad Nawaz
Muhammad Arslan
Mudassar Abbas

GitHub: [Ahmadnawaz27](https://github.com/Ahmadnawaz27)
