from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
import json
from datetime import datetime
import io
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Customer Churn Prediction Platform")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Global variable to store current dataset
current_dataset = None
target_column = None
cleaned_dataset = None
trained_models = {}
openai_client = None

# Request model for data cleaning
class CleaningConfig(BaseModel):
    missing_value_strategy: Literal["drop", "mean"]
    categorical_encoding: Literal["label", "onehot"]
    scaling: Literal["standard", "minmax"]
    
# Request model for training
class TrainingConfig(BaseModel):
    model_name: List[Literal["logistic_regression", "random_forest", "xgboost"]]
    test_size: float = 0.2
    
# Request model for prediction
class PredictionConfig(BaseModel):
    model_name: Literal["logistic_regression", "random_forest", "xgboost"]
    openai_api_key: str


def generate_churn_explanation(customer_data: dict, prediction: int) -> str:
    """
    Generate natural language explanation using OpenAI.
    """
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured. Please add OPENAI_API_KEY to .env file."
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare customer info
        features_str = ", ".join([f"{k}: {v}" for k, v in customer_data.items()])
        churn_status = "likely to churn" if prediction == 1 else "unlikely to churn"
        
        prompt = f"""You are a customer analytics expert. Based on the following customer data, explain why this customer is {churn_status}.

Customer Data: {features_str}

Provide a brief, clear explanation (2-3 sentences) focusing on the key factors that influence this prediction."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer analytics assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API", "status": "running"}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...),target_col: str = Form(...)):
    """
    Upload CSV dataset and specify target column.
    Returns dataset metadata.
    """
    global current_dataset, target_column
    
    try:
        #save uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read CSV
        df = pd.read_csv(file_path)
        
         # Verify target column exists
        if target_col not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_col}' not found in dataset."
            )
        
         # Store globally
        current_dataset = df
        target_column = target_col
        
        # Prepare metadata
        metadata = {
            "filename": file.filename,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "target_column": target_col,
            "columns_info": {},
            "missing_values": {},
            "sample_preview": df.head(5).to_dict(orient="records")
        }
        
        # Column types and missing values
        for col in df.columns:
            metadata["columns_info"][col] = str(df[col].dtype)
            metadata["missing_values"][col] = int(df[col].isnull().sum())
        
        return JSONResponse(content=metadata)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data_cleaning")
def clean_data(config: CleaningConfig):
    """
    Apply data cleaning based on user configuration.
    Returns preview of cleaned dataset.
    """
    global current_dataset, cleaned_dataset, target_column
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    
    try:
        df = current_dataset.copy()
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        Y = df[target_column]
        
        # Initialize preprocessing storage
        label_encoders = {}
        onehot_columns = None
        scaler = None
        le_target = None
        
        # Encode target column if it's categorical
        if Y.dtype == "object":
            le_target = LabelEncoder()
            Y = le_target.fit_transform(Y)
        
        # 1. Handle Missing Values (only 'drop' and 'mean')
        if config.missing_value_strategy == "drop":
            X = X.dropna()
            Y = Y.loc[X.index]
        elif config.missing_value_strategy == "mean":
            numeric_cols = X.select_dtypes(include=['int64','float64']).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
            
        # 2. Encode Categorical Variables (only 'label' and 'onehot')
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if config.categorical_encoding == "label":
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        elif config.categorical_encoding == "onehot":
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            onehot_columns = X.columns.tolist()
            
        # 3. Feature Scaling (only 'standard' and 'minmax')
        numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
        if config.scaling == "standard":
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        elif config.scaling == "minmax":
            scaler = MinMaxScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            
        # Store preprocessing artifacts
        preproc = {
            "categorical_encoding": config.categorical_encoding,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "feature_columns": X.columns.tolist(),
            "onehot_columns": onehot_columns,
            "target_encoder": le_target if Y.dtype == "object" else None
        }
        
        with open("models/preprocessing.pkl", "wb") as pf:
            pickle.dump({
                "preproc": preproc, 
                "label_encoders": label_encoders, 
                "scaler": scaler
            }, pf)
            
        # Combine cleaned features and target
        cleaned_df = X.copy()
        cleaned_df[target_column] = Y
        
        # Store cleaned dataset
        cleaned_dataset = cleaned_df
        
        # Return preview
        response = {
            "message": "Data cleaning completed",
            "shape": {"rows": len(cleaned_df), "columns": len(cleaned_df.columns)},
            "columns": cleaned_df.columns.tolist(),
            "preview": cleaned_df.head(10).to_dict(orient="records"),
            "applied_config": config.dict()
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_model")
def train_model(config: TrainingConfig):
    """
    Train selected models and return evaluation metrics.
    """
    global cleaned_dataset, target_column, trained_models
    
    if cleaned_dataset is None:
        raise HTTPException(status_code=400, detail="No cleaned dataset available. Use /data_cleaning first.")
    
    try:
        # Prepare data
        X = cleaned_dataset.drop(columns=[target_column])
        Y = cleaned_dataset[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=config.test_size, random_state=0
        )
        
        results = {}
        
        # Train each selected model
        for model_name in config.model_name:
            if model_name == "logistic_regression":
                model = LogisticRegression(max_iter=1000, random_state=0)
            elif model_name == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=0)
            elif model_name == "xgboost":
                model = XGBClassifier(random_state=0, eval_metric='logloss')
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}' requested.")
            
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0, pos_label=1)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0, pos_label=1)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0, pos_label=1)
            cm = confusion_matrix(y_test, y_pred)
            
            # Save model
            model_path = f"models/{model_name}.pkl"
            with open(model_path,"wb") as f:
                pickle.dump(model,f)
            
            # Store in memory
            trained_models[model_name] = model
            
            results[model_name] = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm.tolist(),
                "model_saved": model_path
            }
            
        response = {
             "message": "Models trained successfully",
            "results": results,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict")
@app.post("/predict")
async def predict_churn(file: UploadFile = File(...), model_name: str = Form(...)):
    """
    Make predictions on test data and generate GenAI explanations.
    """
    global trained_models, target_column
    
    # Check if OpenAI key is configured
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please add OPENAI_API_KEY to .env file."
        )
    
    # Validate model exists
    if model_name not in trained_models:
        # Try to load from file
        model_path = f"models/{model_name}.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model_name}' not found. Train it first using /train_model"
            )
        with open(model_path, "rb") as f:
            trained_models[model_name] = pickle.load(f)
    
    try:
        # Read test CSV
        test_path = f"uploads/test_{file.filename}"
        with open(test_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        test_df = pd.read_csv(test_path)
        
        # Store original data for explanations
        original_data = test_df.copy()
        
        # Remove target column if present
        if target_column in test_df.columns:
            test_df = test_df.drop(columns=[target_column])
            
        # Load preprocessing artifacts
        preproc_path = "models/preprocessing.pkl"
        if not os.path.exists(preproc_path):
            raise HTTPException(
                status_code=400,
                detail="Preprocessing artifacts not found. Run data cleaning first."
            )
            
        with open(preproc_path, "rb") as pf:
            prep_artifacts = pickle.load(pf)
            
        preproc = prep_artifacts["preproc"]
        label_encoders = prep_artifacts["label_encoders"]
        scaler = prep_artifacts["scaler"]
        
        # Apply preprocessing steps
        # 1. Handle categorical columns
        if preproc["categorical_encoding"] == "label":
            for col in preproc["categorical_cols"]:
                if col in test_df.columns:
                    le = label_encoders.get(col)
                    if le is not None:
                        # Handle unseen categories
                        test_df[col] = test_df[col].astype(str)
                        test_df[col] = test_df[col].map(
                            {c: i for i, c in enumerate(le.classes_)}
                        ).fillna(-1).astype(int)
                        
        elif preproc["categorical_encoding"] == "onehot":
            # Create dummy variables
            test_df = pd.get_dummies(
                test_df, 
                columns=[c for c in preproc["categorical_cols"] if c in test_df.columns],
                drop_first=True
            )
            # Align columns with training data
            for col in preproc["feature_columns"]:
                if col not in test_df.columns:
                    test_df[col] = 0
            test_df = test_df.reindex(columns=preproc["feature_columns"], fill_value=0)
            
        # 2. Apply scaling if used during training
        if scaler is not None:
            numeric_cols = [c for c in preproc["numeric_cols"] if c in test_df.columns]
            if numeric_cols:
                test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
        
        # Get model and make predictions
        model = trained_models[model_name]
        predictions = model.predict(test_df)
        
        # Generate explanations for each row
        results = []
        for idx, (_, row) in enumerate(original_data.iterrows()):
            customer_data = row.to_dict()
            prediction = int(predictions[idx])
            
            # Generate AI explanation
            explanation = generate_churn_explanation(customer_data, prediction)
            
            results.append({
                "row_index": idx,
                "customer_data": customer_data,
                "prediction": prediction,
                "prediction_label": "Churn" if prediction == 1 else "No Churn",
                "explanation": explanation
            })
        
        response = {
            "model_used": model_name,
            "total_predictions": len(predictions),
            "predictions": results
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))