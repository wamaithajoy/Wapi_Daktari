from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from src.api.utils import preprocess_features
app = FastAPI()

# Load the trained models and preprocessor
rf_regressor = joblib.load('../src/api/random_forest_regressor.pkl')
xgb_regressor = joblib.load('../src/api/xgboost_regressor.pkl')
hybrid_regressor = joblib.load('../src/api/hybrid_regressor.pkl')
rf_classifier = joblib.load('../src/api/random_forest_classifier.pkl')
xgb_classifier = joblib.load('../src/api/xgboost_classifier.pkl')
hybrid_classifier = joblib.load('../src/api/hybrid_classifier.pkl')
preprocessor = joblib.load('../src/api/preprocessor.pkl')
label_encoder = joblib.load('../src/api/label_encoder.pkl')

class USSDRequest(BaseModel):
    hospital: str
    department: str
    day_of_week: int
    is_weekend: int
    is_holiday: int
    is_strike_day: int
    time_block: str
    doctors_on_shift: int
    expected_patients: int
    actual_patients: int
    waiting_time_minutes: int
    peak_hour: int
    doctor_available: int
    doctor_arrival_delay: int
    congestion_level: int
    month: int
    day: int
    patient_load_ratio: float
    doctor_patient_ratio: float
    holiday_strike_interaction: int
    expected_walk_ins: int
    emergencies: int
    seasonal_illnesses: int
    public_holidays_events: int
    hour_of_day: int
    day_of_month: int
    quarter: int
    season: str
    previous_day_patients: int
    previous_week_patients: int
    previous_month_patients: int
    temperature: int
    humidity: int
    rainfall: int
    school_holidays: int
    national_events: int
    average_waiting_time_last_week: int
    average_patients_last_month: int
    previous_day_waiting_time: int
    previous_week_waiting_time: int
    previous_month_waiting_time: int
    doctors_on_shift_expected_patients: int
    doctor_patient_ratio_congestion_level: int
    flu_season: int
    malaria_season: int

@app.post('/ussd')
def ussd_response(request: USSDRequest):
    features = [
        request.day_of_week, request.is_weekend, request.is_holiday, request.is_strike_day,
        request.department, request.time_block, request.doctors_on_shift, request.expected_patients,
        request.actual_patients, request.waiting_time_minutes, request.peak_hour, request.doctor_available,
        request.doctor_arrival_delay, request.congestion_level, request.month, request.day,
        request.patient_load_ratio, request.doctor_patient_ratio, request.holiday_strike_interaction,
        request.expected_walk_ins, request.emergencies, request.seasonal_illnesses, request.public_holidays_events,
        request.hour_of_day, request.day_of_month, request.quarter, request.season, request.previous_day_patients,
        request.previous_week_patients, request.previous_month_patients, request.temperature, request.humidity, request.rainfall,
        request.school_holidays, request.national_events, request.average_waiting_time_last_week, request.average_patients_last_month,
        request.previous_day_waiting_time, request.previous_week_waiting_time, request.previous_month_waiting_time,
        request.doctors_on_shift_expected_patients, request.doctor_patient_ratio_congestion_level,
        request.flu_season, request.malaria_season
    ]

    features_scaled = preprocessor.transform(np.array(features).reshape(1, -1))
    prediction_rf_reg = rf_regressor.predict(features_scaled)
    prediction_xgb_reg = xgb_regressor.predict(features_scaled)
    prediction_hybrid_reg = hybrid_regressor.predict(features_scaled)
    prediction_rf_class = rf_classifier.predict(features_scaled)
    prediction_xgb_class = xgb_classifier.predict(features_scaled)
    prediction_hybrid_class = hybrid_classifier.predict(features_scaled)

    response = f"Best time to visit: {prediction_rf_reg[0]:.2f} mins (RF), {prediction_xgb_reg[0]:.2f} mins (XGB), {prediction_hybrid_reg[0]:.2f} mins (Hybrid). Congestion level: {label_encoder.inverse_transform(prediction_rf_class)[0]}, {label_encoder.inverse_transform(prediction_xgb_class)[0]}, {label_encoder.inverse_transform(prediction_hybrid_class)[0]}."
    return {"response": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)