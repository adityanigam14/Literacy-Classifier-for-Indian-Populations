from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

class InputData(BaseModel):
    social_group: str
    rural_urban: str
    state: str
    gender: str
    age: int
    internet_access: str
    computer_access: str
    marital_status: str

# Load the trained model in the global scope
try:
    model = tf.keras.models.load_model('literacy_classifier.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model.")

# Define a function to preprocess the input data so it matches the format that the model expects

def preprocess_input(data: dict):
    
    model_columns = [
    'onehot__Social Group_1', 'onehot__Social Group_2',
    'onehot__Social Group_3', 'onehot__Social Group_9', 'onehot__Rural/Urban_1',
    'onehot__Rural/Urban_2', 'onehot__Gender_1', 'onehot__Gender_2',
    'onehot__Marital Status_1', 'onehot__Marital Status_2',
    'onehot__Marital Status_3', 'onehot__Digital Access_0',
    'onehot__Digital Access_1', 'onehot__Digital Access_2',
    'onehot__Age Bracket_18-35', 'onehot__Age Bracket_35-60',
    'onehot__Age Bracket_<18', 'onehot__Age Bracket_>60',
    'onehot__Region_Central India', 'onehot__Region_East India',
    'onehot__Region_North India', 'onehot__Region_Northeast India',
    'onehot__Region_South India', 'onehot__Region_Union Territories',
    'onehot__Region_West India'
    ]

    # Mapping for state to region
    state_to_region = {
        "Jammu & Kashmir": "North India",
        "Himachal Pradesh": "North India",
        "Punjab": "North India",
        "Chandigarh": "North India",
        "Uttarakhand": "North India",
        "Haryana": "North India",
        "Delhi": "North India",
        "Rajasthan": "North India",
        "Uttar Pradesh": "North India",
        "Bihar": "East India",
        "Sikkim": "Northeast India",
        "Arunachal Pradesh": "Northeast India",
        "Nagaland": "Northeast India",
        "Manipur": "Northeast India",
        "Mizoram": "Northeast India",
        "Tripura": "Northeast India",
        "Meghalaya": "Northeast India",
        "Assam": "Northeast India",
        "West Bengal": "East India",
        "Jharkhand": "East India",
        "Odisha": "East India",
        "Chhattisgarh": "Central India",
        "Madhya Pradesh": "Central India",
        "Gujarat": "West India",
        "Daman & Diu": "Union Territories",
        "Dadara and Nagar Haveli": "Union Territories",
        "Maharashtra": "West India",
        "Andhra Pradesh": "South India",
        "Karnataka": "South India",
        "Goa": "West India",
        "Lakshadweep": "Union Territories",
        "Kerala": "South India",
        "Tamil Nadu": "South India",
        "Pondicherry": "South India",
        "Andaman and Nicobar Islands": "Union Territories",
        "Telangana": "South India",
    }

    social_group_mapping = {
        "Scheduled Tribes": 1,
        "Scheduled Castes": 2,
        "Other Backward Classes": 3,
        "Others": 9
    }
    marital_status_mapping = {
        "Single": 1,
        "Married": 2,
        "Widowed": 3
    }
    
    rural_urban_mapping = {
        "Rural": 1,
        "Urban": 2
    }

    gender_mapping = {
        "Male": 1,
        "Female": 2
    }
    
    def bin_age(age):
        if age < 18:
            return '<18'
        elif 18 <= age < 35:
            return '18-35'
        elif 35 <= age < 60:
            return '35-60'
        else:
            return '>60'

    digital_access = 0
    if data.get("internet_access") == "Yes":
        digital_access += 1
    if data.get("computer_access") == "Yes":
        digital_access += 1

    region = state_to_region.get(data.get("state", ""), "Unknown")
    age_bracket = bin_age(data.get("age", 0))
    social_group = social_group_mapping.get(data.get("social_group", ""))
    marital_status = marital_status_mapping.get(data.get("marital_status", ""))
    rural_urban = rural_urban_mapping.get(data.get("rural_urban", ""), "Unknown")
    gender = gender_mapping.get(data.get("gender", ""), "Unknown")

    categorical_features = {
        "social_group": social_group,
        "rural_urban": rural_urban,
        "gender": gender,
        "marital_status": marital_status,
        "digital_access": digital_access, 
        "age_bracket": age_bracket,
        "region": region
    }
    
    # Create a DataFrame for one-hot encoding
    df = pd.DataFrame([categorical_features])
    print("Raw Categorical Features:", df)

    # Apply one-hot encoding with consistent prefixes
    df_encoded = pd.get_dummies(
        df,
        columns=["social_group", "rural_urban", "gender", "marital_status", "digital_access", "age_bracket", "region"],
        prefix=["onehot__Social Group", "onehot__Rural/Urban", "onehot__Gender", "onehot__Marital Status", "onehot__Digital Access", "onehot__Age Bracket", "onehot__Region"]
    )

    # Align columns with the model's expected input columns
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    return df_encoded.values.astype(np.float32)

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Step 1: Preprocess the input data
        processed_input = preprocess_input(data.dict())
        print(f"Processed input: {processed_input}")
        print(f"Processed input shape: {processed_input.shape}")

        # Step 2: Test the model prediction
        prediction = model.predict(processed_input)
        print(f"Raw prediction: {prediction}")

        # Extract the probability and apply threshold
        raw_prediction = float(prediction[0][0])
        threshold = 0.30  # Use the same threshold as during evaluation
        binary_prediction = 1 if raw_prediction > threshold else 0
        literacy_status = "Literate" if binary_prediction == 1 else "Illiterate"

        return {
            "probability": raw_prediction,
            "class": binary_prediction,
            "status": literacy_status,
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Literacy Classifier API! Use /predict to make predictions."}

# Run the FastAPI app with uvicorn (uvicorn app:app --reload)