# Wapi Daktari: Public Healthcare Crowd-Management System

## Problem Statement

In Kenya, many citizens visiting public hospitals face significant challenges, among those challenges that are common are;

- Uncertainity about doctor availability.
- Unknown waiting times.
- Lack of Information of whether they'll be seen at all.

These issues affect ordinary kenyans who cant afford private healthcare or appointment systems. THis uncertainity leads to:

- Wasted times for patients
- Discouragement from seeking medical care.
- Random overloading of hospital resources.

## Solution: Wapi Daktari

Wapi Daktari is a public healthcare crowd-management and awareness system designed to empower ordinary kenyans, particularly walk-in patients at public hospitals. The system predicts the best time to visit a hospital based on historical patterns of doctor availability, patient traffic, and waiting times.

Wapi Daktari Answers the question?

When is the best time for me to go to the hospiotal today?

## Key Features

- Works via USSD, ensuring accessibility for all mobile phone users without requiring smartphones or Internet access.
- Provide Instant, data-driven predictions for hospital visits.
- Available in English and Kiswahili to cater to a wider audience.
- Multiple hospital and departments support across Kenya.
- Flexinble: Allows users to check predictions for today, tomorrow, day after tomorrow, or a specific date.

## How it Works

- Users dial *384*43371# on any mobile phone.
- They select their preferred language ( English or Kiswahili).
- Users choose their hospital and department of interest.
- They select the date for their hospital visit.
- The system analyzes real hospital data using machine learning models to return:

1. Best time to visit the hospital
2. Estimated waiting time
3. Predicted congestion level

All responses are sent instantly via USSD, requiring no internet, app, or smartphone.

## Technologies Used

FastAPI – API backend for handling USSD requests and predictions

Python – Core language for ML and backend logic

Pandas – Data manipulation and analysis

Joblib – Model loading

Scikit-learn – Preprocessing, training, and evaluation

NumPy – Numerical computations

Datetime – Date and time manipulation

Africastalking – USSD integration for mobile interaction

Streamlit – Interactive dashboard for hospitals and stakeholders

## Machine Learning Models

Wapi Daktari employs six different machine learning models to provide accurate predictions:

1. Random Forest Regressor: Predicts waiting times
2. XGBoost Regressor: Alternative model for waiting time prediction.
3. Hybrid Regressor: Ensemble model combining Random Forest Regrossor and XGBoost for waiting time prediction.
4. Random Forest Classifier: Predicts congestion levels.
5. XGBoost Classifier: Alternative model for congestion level prediction.
6. Hybrid Classifier: Ensemble model combining Random Forest Classifier and XGBoost for congestion level prediction.

## Model Purpose

- Regressors: Predict continous values (waiting time in minutes)
- Classifiers: Predict categorical values (congestion levels ie; Low, Medium, High)
- Hybrid Models: Combine the strengths of both Random Forest and XGBoost to Improve prediction accuracy.

## USSD Application

The USSD application serves as the user interface for Wapi Daktari. It's designed to be simple, intuitive, and accessible on any mobile phone. The application flow is as follows:

1.Language Selection
2.Main Menu (Check best time or Change language)
3.Hospital Selection
4.Department Selection
5.Date Selection
6.Result Display

The application handles user inputs, manages the conversations flow, and integrates with the machine learning models to provide predictions.

## Future Enhancements

- Real-time hospital API integrations

- SMS-based appointment reminders

- Region-specific awareness alerts (Eg; strikes, outbreaks, holidays)

- Expanded Language Support: Add more local languages to increase accessibility.

- Toggle for Daily vs Weekly Data View: Allowing users to see predictions for an entire week.

- Feedback System.

- Mobile APP Version: Develop a complimentary smartphone app for users who prefer a graphical interface.

## Credits

Built by: Joy Wamaitha

Email: joyywamaitha@gmail.com
