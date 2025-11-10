# 🤖 Customer Churn Prediction Platform

A production-ready FastAPI application that combines **Machine Learning** and **Generative AI** to predict customer churn and provide natural language explanations for each prediction.

---

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [API Endpoints](#api-endpoints)
- [Complete Workflow Example](#complete-workflow-example)
- [Testing with Sample Data](#testing-with-sample-data)
- [GenAI Integration](#genai-integration)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **📤 Dataset Upload**: Upload CSV files with flexible target column selection
- **🧹 Configurable Data Cleaning**: 
  - Multiple missing value strategies (drop, mean, median, mode)
  - Categorical encoding (label encoding, one-hot encoding)
  - Feature scaling (standard scaler, min-max scaler)
- **🎯 Multi-Model Training**: 
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- **📊 Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **🤖 AI-Powered Explanations**: Natural language explanations using OpenAI GPT-3.5
- **🔒 Secure API Key Management**: Environment-based configuration

---

## 🛠 Tech Stack

- **Framework**: FastAPI 0.104.1
- **ML Libraries**: scikit-learn, XGBoost
- **GenAI**: OpenAI API (GPT-3.5-turbo)
- **Data Processing**: Pandas, NumPy
- **Server**: Uvicorn

---

## 📁 Project Structure

```
churn_prediction/
├── main.py                 # FastAPI application with all endpoints
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── .gitignore             # Git ignore file
├── README.md              # This file
├── models/                # Trained model files (.pkl)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── uploads/               # Uploaded CSV files
│   ├── train_data.csv
│   └── test_data.csv
└── sample_data/           # Sample datasets (optional)
    └── customer_churn.csv
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Step 1: Clone/Download the Project

```bash
mkdir churn_prediction
cd churn_prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
scikit-learn==1.3.2
xgboost==2.0.2
python-multipart==0.0.6
openai==1.3.7
numpy==1.26.2
python-dotenv==1.0.0
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

⚠️ **Important**: Never commit your `.env` file to version control!

Create/update `.gitignore`:
```
.env
venv/
__pycache__/
*.pyc
uploads/*.csv
models/*.pkl
```

### Step 5: Create Required Directories

The application creates these automatically, but you can create them manually:

```bash
mkdir models
mkdir uploads
```

### Step 6: Run the Application

```bash
uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

---

## 📡 API Endpoints

### 🌐 Base URL: `http://localhost:8000`

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

### 1️⃣ **POST** `/upload` - Upload Dataset

Upload your training dataset and specify the target column.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: CSV file (required)
  - `target_col`: Name of target column, e.g., "Churn" (required)

**Response:**
```json
{
  "filename": "customer_data.csv",
  "total_rows": 7043,
  "total_columns": 21,
  "target_column": "Churn",
  "columns_info": {
    "customerID": "object",
    "tenure": "int64",
    "MonthlyCharges": "float64",
    "Churn": "object"
  },
  "missing_values": {
    "customerID": 0,
    "tenure": 0,
    "TotalCharges": 11
  },
  "sample_preview": [...]
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@customer_data.csv" \
  -F "target_col=Churn"
```

---

### 2️⃣ **POST** `/data_cleaning` - Clean & Preprocess Data

Apply preprocessing transformations to your dataset.

**Request:**
- **Content-Type**: `application/json`

**Request Body:**
```json
{
  "missing_value_strategy": "mean",
  "categorical_encoding": "label",
  "scaling": "standard"
}
```

**Options:**
- `missing_value_strategy`: 
  - `"drop"` - Remove rows with missing values
  - `"mean"` - Fill numeric columns with mean
  - `"median"` - Fill numeric columns with median
  - `"mode"` - Fill all columns with mode
  
- `categorical_encoding`:
  - `"label"` - Label encoding (A=0, B=1, C=2)
  - `"onehot"` - One-hot encoding (creates dummy variables)
  
- `scaling`:
  - `"standard"` - StandardScaler (mean=0, std=1)
  - `"minmax"` - MinMaxScaler (range 0-1)
  - `"none"` - No scaling

**Response:**
```json
{
  "message": "Data cleaning completed",
  "shape": {
    "rows": 7032,
    "columns": 21
  },
  "columns": ["tenure", "MonthlyCharges", "TotalCharges", ...],
  "preview": [...],
  "applied_config": {
    "missing_value_strategy": "mean",
    "categorical_encoding": "label",
    "scaling": "standard"
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/data_cleaning" \
  -H "Content-Type: application/json" \
  -d '{
    "missing_value_strategy": "mean",
    "categorical_encoding": "label",
    "scaling": "standard"
  }'
```

---

### 3️⃣ **POST** `/train_model` - Train ML Models

Train one or multiple machine learning models.

**Request:**
- **Content-Type**: `application/json`

**Request Body:**
```json
{
  "models": ["logistic_regression", "random_forest", "xgboost"],
  "test_size": 0.2
}
```

**Options:**
- `models`: Array of model names (choose one or more)
  - `"logistic_regression"`
  - `"random_forest"`
  - `"xgboost"`
- `test_size`: Fraction for test split (default: 0.2 = 20%)

**Response:**
```json
{
  "message": "Models trained successfully",
  "results": {
    "logistic_regression": {
      "accuracy": 0.8045,
      "precision": 0.6721,
      "recall": 0.5512,
      "f1_score": 0.6055,
      "confusion_matrix": [[965, 68], [197, 241]],
      "model_saved": "models/logistic_regression.pkl"
    },
    "random_forest": {
      "accuracy": 0.7923,
      "precision": 0.6354,
      "recall": 0.4732,
      "f1_score": 0.5423,
      "confusion_matrix": [[1002, 31], [231, 207]],
      "model_saved": "models/random_forest.pkl"
    }
  },
  "train_size": 5625,
  "test_size": 1407
}
```

**Metrics Explanation:**
- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted churns, how many were correct
- **Recall**: Of actual churns, how many were caught
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: `[[TN, FP], [FN, TP]]`

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/train_model" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["random_forest", "xgboost"],
    "test_size": 0.2
  }'
```

---

### 4️⃣ **POST** `/predict` - Make Predictions with AI Explanations

Generate predictions on new data with natural language explanations.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Test CSV file (required)
  - `model_name`: Model to use for predictions (required)

**Response:**
```json
{
  "model_used": "random_forest",
  "total_predictions": 10,
  "predictions": [
    {
      "row_index": 0,
      "customer_data": {
        "customerID": "7590-VHVEG",
        "tenure": 1,
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "Contract": "Month-to-month"
      },
      "prediction": 0,
      "prediction_label": "No Churn",
      "explanation": "Customer is unlikely to churn despite having a month-to-month contract because their monthly charges are relatively low at $29.85, indicating good value perception. Their short tenure of 1 month is typical for new customers who are still evaluating the service."
    },
    {
      "row_index": 1,
      "customer_data": {
        "customerID": "5575-GNVDE",
        "tenure": 34,
        "MonthlyCharges": 56.95,
        "TotalCharges": 1889.5,
        "Contract": "One year"
      },
      "prediction": 1,
      "prediction_label": "Churn",
      "explanation": "Customer is likely to churn due to their one-year contract nearing renewal (34 months tenure). The relatively high monthly charges of $56.95 combined with the contract expiration creates a critical decision point where customers often evaluate competitors."
    }
  ]
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_customers.csv" \
  -F "model_name=random_forest"
```

---

## 🔄 Complete Workflow Example

Here's a complete end-to-end example using sample data:

```bash
# Step 1: Upload training dataset
curl -X POST "http://localhost:8000/upload" \
  -F "file=@customer_churn_train.csv" \
  -F "target_col=Churn"

# Step 2: Clean the data
curl -X POST "http://localhost:8000/data_cleaning" \
  -H "Content-Type: application/json" \
  -d '{
    "missing_value_strategy": "median",
    "categorical_encoding": "label",
    "scaling": "standard"
  }'

# Step 3: Train multiple models
curl -X POST "http://localhost:8000/train_model" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["logistic_regression", "random_forest", "xgboost"],
    "test_size": 0.25
  }'

# Step 4: Make predictions on new data
curl -X POST "http://localhost:8000/predict" \
  -F "file=@customer_churn_test.csv" \
  -F "model_name=xgboost"
```

---

## 🧪 Testing with Sample Data

### Using Python Requests

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Upload dataset
with open("customer_data.csv", "rb") as f:
    files = {"file": f}
    data = {"target_col": "Churn"}
    response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
    print(response.json())

# 2. Clean data
cleaning_config = {
    "missing_value_strategy": "mean",
    "categorical_encoding": "label",
    "scaling": "standard"
}
response = requests.post(f"{BASE_URL}/data_cleaning", json=cleaning_config)
print(response.json())

# 3. Train models
training_config = {
    "models": ["random_forest"],
    "test_size": 0.2
}
response = requests.post(f"{BASE_URL}/train_model", json=training_config)
print(response.json())

# 4. Make predictions
with open("test_data.csv", "rb") as f:
    files = {"file": f}
    data = {"model_name": "random_forest"}
    response = requests.post(f"{BASE_URL}/predict", files=files, data=data)
    print(json.dumps(response.json(), indent=2))

## 🤖 GenAI Integration

### How AI Explanations Work

The `/predict` endpoint uses OpenAI's GPT-3.5-turbo to generate human-readable explanations:

**Prompt Template:**
```
System: You are a helpful customer analytics assistant.

User: You are a customer analytics expert. Based on the following 
customer data, explain why this customer is [likely/unlikely] to churn.

Customer Data: tenure: 12, MonthlyCharges: 89.5, Contract: Month-to-month

Provide a brief, clear explanation (2-3 sentences) focusing on 
the key factors that influence this prediction.
```

**API Configuration:**
- **Model**: `gpt-3.5-turbo`
- **Max Tokens**: 150
- **Temperature**: 0.7
- **Purpose**: Generate concise, actionable insights

### Customizing AI Responses

To modify the explanation style, edit the `generate_churn_explanation()` function in `main.py`:

```python
# Change the prompt for different explanation styles
prompt = f"""You are a customer retention specialist...
[Your custom prompt here]
"""

# Adjust temperature for creativity
temperature=0.5  # More focused (0.0-1.0)
```

---

## 🔒 Security Best Practices

### Environment Variables
✅ Store API keys in `.env` file  
✅ Never commit `.env` to version control  
✅ Use `.gitignore` to exclude sensitive files  
❌ Never hardcode API keys in source code  

### API Security
- Use HTTPS in production
- Implement rate limiting
- Add authentication middleware
- Validate file uploads (size, type)
- Sanitize user inputs

### Deployment Checklist
```bash
# Before deploying:
1. Set environment variables on server
2. Use production ASGI server (gunicorn + uvicorn)
3. Enable CORS for specific domains only
4. Implement logging and monitoring
5. Set up SSL/TLS certificates
```

---

## 📝 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

**Built with ❤️ using FastAPI, scikit-learn, and OpenAI**

