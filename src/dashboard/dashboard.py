# src/dashboard/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('data/wapi_daktari_healthcare_dataset.csv')

# Dashboard title
st.title('Healthcare Dashboard')

# Sidebar filters
hospital = st.sidebar.selectbox('Select Hospital', df['hospital_name'].unique())
department = st.sidebar.selectbox('Select Department', df['department'].unique())

# Filter data
filtered_data = df[(df['hospital_name'] == hospital) & (df['department'] == department)]

# Visualizations
st.subheader('Waiting Time Distribution')
fig = px.histogram(filtered_data, x='waiting_time_minutes', nbins=20)
st.plotly_chart(fig)

st.subheader('Expected vs Actual Patients')
fig = px.scatter(filtered_data, x='expected_patients', y='actual_patients')
st.plotly_chart(fig)

st.subheader('Walk-In Traffic')
fig = px.bar(filtered_data, x='time_block', y='expected_walk_ins', color='day_of_week')
st.plotly_chart(fig)

st.subheader('Emergencies and Seasonal Illnesses')
fig = px.line(filtered_data, x='date', y=['emergencies', 'seasonal_illnesses'])
st.plotly_chart(fig)

st.subheader('Public Holidays and Events')
fig = px.bar(filtered_data, x='date', y='public_holidays_events')
st.plotly_chart(fig)

st.subheader('Weather Impact')
fig = px.scatter(filtered_data, x='temperature', y='actual_patients', color='humidity')
st.plotly_chart(fig)

st.subheader('Patient Load Ratio')
fig = px.histogram(filtered_data, x='patient_load_ratio', nbins=20)
st.plotly_chart(fig)

st.subheader('Doctor Patient Ratio')
fig = px.histogram(filtered_data, x='doctor_patient_ratio', nbins=20)
st.plotly_chart(fig)

st.subheader('Congestion Level')
fig = px.histogram(filtered_data, x='congestion_level')
st.plotly_chart(fig)
