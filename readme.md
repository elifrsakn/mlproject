
# Student Performance Prediction – End-to-End ML Project

This repository contains a fully modular and production-ready end-to-end machine learning pipeline for predicting student performance using the `stud.csv` dataset. The architecture is designed following software engineering best practices, with modular components for data ingestion, preprocessing, model training, evaluation, logging, and deployment setup.

---

## 🚀 Project Overview

📌 **Goal**: Predict student performance based on features such as demographics, study habits, and other related variables.  
💡 **Model**: CatBoostClassifier – chosen for its high accuracy and native support for categorical features.  
🛠 **Deployment**: Flask-based API with AWS deployment configurations (Elastic Beanstalk-ready).

---

## 📁 Directory Structure
mlproject/
│
├── .bextensions/ # AWS Elastic Beanstalk deployment config
├── artifacts/ # Stored artifacts (processed data, model.pkl, etc.)
├── catboost_info/ # CatBoost logs (training progress, errors, etc.)
│ ├── learn/
│ ├── tmp/
│ ├── catboost_training.json
│ ├── learn_error.tsv
│ └── time_left.tsv
│
├── logs/ # Custom logging output
├── notebook/ # Jupyter notebooks for EDA / experimentation
│ └── data/ # Contains raw stud.csv
│
├── src/ # Source code
│ ├── components/ # Data ingestion, transformation, model trainer, evaluator
│ ├── pipeline/ # Prediction pipeline for serving new data
│ ├── templates/ # HTML templates for Flask UI (if any)
│ ├── app.py # Main app logic
│ ├── exception.py # Custom exception handling
│ ├── logger.py # Custom logging
│ ├── utils.py # Helper functions
│ └── init.py
│
├── application.py # Flask entry point for inference API
├── requirements.txt # Python dependencies
├── setup.py # For packaging the module
├── .gitignore
└── readme.md


---

## 🛠️ Features & Highlights

- ✅ Modularized pipeline using OOP and Python best practices
- 📊 EDA-ready Jupyter notebooks
- 🧪 CatBoost model with early stopping and evaluation
- 🧱 Robust logging and exception handling
- 🌐 Flask-based deployment setup
- ☁️ AWS deployment config (Elastic Beanstalk `.bextensions/`)
- 🔐 Scalable structure for CI/CD and MLOps workflows

---

## 🧪 ML Pipeline Flow

1. **Data Ingestion**  
   → Loads and splits the raw dataset from `notebook/data/stud.csv`  
2. **Data Transformation**  
   → Applies encoding, missing value handling, and feature selection  
3. **Model Training**  
   → Trains a CatBoost model and evaluates performance  
4. **Model Saving**  
   → Stores the model in the `artifacts/` directory  
5. **Prediction Pipeline**  
   → Loads the model and predicts on new user input  
6. **Deployment (optional)**  
   → Ready-to-deploy Flask API defined in `application.py`

---

## ▶️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/elifrsakn/mlproject.git
cd mlproject

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python application.py



