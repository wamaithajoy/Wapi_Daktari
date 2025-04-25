# Wapi Daktari: Public Healthcare Crowd-Management System

## Project Overview

Wapi Daktari is a public healthcare crowd-management and awareness system designed to help ordinary Kenyans, especially walk-in patients at public hospitals, by predicting the best time to visit a hospital based on past patterns of doctor availability, patient traffic, and waiting times.

The system provides real-time predictions via a simple USSD interface, ensuring that patients receive timely and actionable insights without the need for an app or account.

## Problem

Most Kenyans visiting public hospitals;

- Don’t know if the doctor is available,
- How long the wait will be, or if they'll even be seen.
- They cant afford private healthcare or appointment systems.

This uncertainty wastes time, discourages care-seeking, and overloads hospitals randomly. There’s no access to real-time hospital insights especially for low-income patients without smartphones or internet access.

## Objective

To empower walk-in patients at public hospitals with data-backed predictions on the best time to visit via a simple USSD interface that works on any phone. The goal is to reduce wasted time, manage hospital congestion, and improve access to care without requiring logins or apps.

Wapi Daktari Answers the question?

When is the best time for me to go to the hospiotal today?

## Technologies Used

FastAPI – API backend for handling USSD requests and predictions

Python – Core language for ML and backend logic

Joblib – Model loading

Scikit-learn – Preprocessing, training, and evaluation

NumPy – Numerical computations

Pydantic – Data validation and serialization

Africastalking – USSD integration for mobile interaction

Streamlit – Interactive dashboard for hospitals and stakeholders

## How It Works

For Patients (USSD):
A user dials the USSD code on any mobile phone.

They select their hospital and department.

The system analyzes real hospital data using ML models to return:

- Best time to visit

- Estimated waiting time

- Predicted congestion level

All responses are sent instantly via USSD—no internet, app, or smartphone needed.

For Hospitals & Stakeholders (Streamlit Dashboard):
Access the secure Streamlit Dashboard from a browser.

Filter by hospital, department, and date range.

Visualize trends in:

Patient load & flow

Doctor availability

Average waiting times

Prediction accuracy

Upload new CSVs or manually input new data to simulate scenarios or feed future models.
