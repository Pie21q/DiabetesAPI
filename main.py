from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model (which you mentioned already includes the encoding pipeline)
model = joblib.load("model.pkl")

class AssessmentInput(BaseModel):
    age: float
    alcohol_consumption_per_week: float
    physical_activity_minutes_per_week: float
    diet_score: float
    sleep_hours_per_day: float
    screen_time_hours_per_day: float
    bmi: float
    waist_to_hip_ratio: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float
    cholesterol_total: float
    hdl_cholesterol: float
    ldl_cholesterol: float
    triglycerides: float
    gender: str
    ethnicity: str
    education_level: str
    income_level: str
    smoking_status: str
    employment_status: str
    family_history_diabetes: int
    hypertension_history: int
    cardiovascular_history: int

@app.post("/predict")
def predict(data: AssessmentInput):
    # 1. Convert the Pydantic object to a dictionary
    input_dict = data.dict()

    # 2. Convert to DataFrame (Pipeline needs a DF to recognize column names)
    df = pd.DataFrame([input_dict])

    # 3. Ensure the columns are in the EXACT order of your test.csv
    column_order = [
        'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
        'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 
        'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 
        'gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 
        'employment_status', 'family_history_diabetes', 'hypertension_history', 
        'cardiovascular_history'
    ]
    df = df[column_order]

    # 4. Pass the raw strings/numbers directly into your pipeline
    prediction = model.predict(df)
    
    return {"prediction": prediction.tolist()}