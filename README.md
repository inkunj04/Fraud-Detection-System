# 🛡️ Fraud Detection System

A powerful, real-time fraud detection web application that uses advanced machine learning to analyze transactions and predict fraudulent activities. This project integrates an end-to-end ML pipeline with an interactive Streamlit dashboard, delivering a practical and robust financial security tool.

---

## 📌 Table of Contents

- [🛡️ Fraud Detection System](#️-fraud-detection-system)
  - [📌 Table of Contents](#-table-of-contents)
  - [🔍 Project Overview](#-project-overview)
  - [🚀 Features](#-features)
  - [📊 Model Details](#-model-details)
  - [🖥️ Streamlit App Screens](#️-streamlit-app-screens)
  - [📁 Project Structure](#-project-structure)
  - [📚 Dataset Info](#-dataset-info)
  - [🧠 What Makes It Stand Out](#-what-makes-it-stand-out)

---

## 🔍 Project Overview

This system identifies and classifies potential fraudulent transactions using a trained **XGBoost machine learning model**. The interactive dashboard allows users to upload data, view fraud analytics, and simulate real-time detection scenarios — ideal for both educational and practical use in financial systems.

---

## 🚀 Features

| Feature                               | Description |
|--------------------------------------|-------------|
| 📁 CSV Upload Support                | Upload your own transaction CSV file. |
| 🧹 Smart Data Preprocessing         | Auto-handles encoding and column cleaning. |
| 🤖 ML Model Integration              | Uses a pre-trained XGBoost model. |
| 📊 Real-Time Summary Dashboard       | Displays key metrics and quick insights. |
| 📈 Performance Evaluation            | Confusion Matrix, Classification Report, and ROC Curve. |
| 🗺️ Location-Based Fraud Heatmap      | Visualizes fraud concentration by location. |
| 🕒 Real-Time Transaction Simulation  | Simulates and classifies transactions one-by-one. |
| 🧾 Downloadable Reports              | Save prediction results as CSV. |
| 🔐 Risk Level Categorization         | Classifies each prediction into Low, Medium, or High Risk. |

---

## 📊 Model Details

- **Model Used:** XGBoost Classifier  
- **Training:** Pre-trained on a dataset of anonymized financial transactions  
- **Label:** `Fraud_Label` (0 = Legit, 1 = Fraud)  
- **Encoding:** Label Encoding for object features  
- **Output:**
  - `Fraud_Prediction`: Binary classification
  - `Fraud_Probability`: Prediction confidence
  - `Risk_Level`: Categorized into Low, Medium, or High

---

## 🖥️ Streamlit App Screens

| Module                         | Description |
|-------------------------------|-------------|
| **Real-Time Dashboard**       | Live metrics on frauds, amounts, and more |
| **Data Preprocessing Preview**| Displays cleaned data post-encoding |
| **Model Prediction**          | Runs ML model on uploaded data |
| **Model Performance**         | Shows Confusion Matrix, ROC Curve, and metrics |
| **Fraud Heatmap**             | Visual view of fraud by location (if available) |
| **Transaction Simulation**    | Streamed predictions in real-time |

---

## 📁 Project Structure

| **File / Folder**                       | **Description**                                                  |
|----------------------------------------|------------------------------------------------------------------|
| `streamlit_app.py`                     | Main Streamlit app file that runs the UI                         |
| `model/fraud_detection_xgb_model.pkl`  | Pre-trained XGBoost model used for fraud prediction              |
| `.gitignore`                           | Prevents large files (like `.csv`, `.pkl`) from being pushed     |
| `README.md`                            | Complete project documentation (this file)                       |
| `requirements.txt` *(optional)*        | Python libraries required to run the project                     |
| `data/` *(optional)*                   | Folder to hold user-uploaded CSVs or input datasets              |
| `output/` *(optional)*                 | Folder to store prediction results or downloadable files         |

---

## 📚 Dataset Info

The model is trained on a **synthetic financial transaction dataset** designed to simulate real-world mobile money transactions, such as those performed via M-Pesa or PayTM.

- **Source**: [Kaggle - PaySim Simulation Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Type**: Synthetic (modeled from real transaction patterns)
- **Total Records**: ~6 million transactions
- **Features Include**:
  - `amount`: Transaction amount
  - `oldbalanceOrg`, `newbalanceOrig`: Sender's balance before and after
  - `oldbalanceDest`, `newbalanceDest`: Receiver's balance before and after
  - `type`: Transaction type (e.g., CASH_OUT, PAYMENT)
  - `isFraud`: Ground truth fraud label (1 = fraud, 0 = legit)

⚠️ **Note**: Large files like `creditcard.csv` (~143 MB) are excluded due to GitHub's 100MB limit. Use `.gitignore` to skip tracking them or upload manually if needed.

---

## 🧠 What Makes It Stand Out

| **Feature**                        | **Why It’s Unique**                                                                 |
|-----------------------------------|--------------------------------------------------------------------------------------|
| Real-Time Transaction Simulation  | Mimics live fraud detection in a production-like scenario using time-delayed logic  |
| Fraud Heatmap by Location         | Displays fraud cases visually by geographic location (if available in the dataset)  |
| Multi-tab Streamlit Interface     | Clean UX with separate tabs for prediction, evaluation, and data visualization      |
| Risk Level Bucketing              | Goes beyond binary classification by categorizing predictions into Low/Med/High     |
| One-Click CSV Report Export       | Allows users to download predictions and risk analysis in one click                 |
| In-App Model Evaluation           | Includes real-time Confusion Matrix, ROC Curve, and classification metrics          |
| Plug-and-Play Dataset Upload      | Upload any valid CSV and get instant predictions without changing the code          |
