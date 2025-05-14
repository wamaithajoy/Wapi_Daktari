import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import os

st.set_page_config(layout="wide")
st.title('📊 Wapi Daktari Healthcare Dashboard')

# Sidebar Upload
st.sidebar.header("Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload healthcare data CSV", type="csv")

# Load default dataset or uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'wapi_daktari_healthcare_dataset.csv')
    df = pd.read_csv(file_path)

# Ensure date format
df['date'] = pd.to_datetime(df['date'])

# Load preprocessor and encoder
preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'preprocessor.pkl')
label_encoder_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'label_encoder.pkl')

preprocessor = joblib.load(preprocessor_path)
label_encoder = joblib.load(label_encoder_path)

# Sidebar filters
hospital = st.sidebar.selectbox('Select Hospital', df['hospital_name'].unique())
department = st.sidebar.selectbox('Select Department', df['department'].unique())

min_date = df['date'].min().date()
max_date = df['date'].max().date()
default_date = min_date

date_picker = st.sidebar.date_input('Select Date', value=default_date, min_value=min_date, max_value=max_date)

filtered_data = df[
    (df['hospital_name'] == hospital) &
    (df['department'] == department) &
    (df['date'].dt.date == date_picker)
]

st.write(f"### Showing data for **{hospital} – {department}** on **{date_picker}**")
st.write(f"Total rows: **{filtered_data.shape[0]}**")

if filtered_data.empty:
    st.warning("No data for this hospital, department, and date. Try a different filter.")
    st.stop()

with st.expander("View Filtered Data"):
    st.dataframe(filtered_data)

if st.sidebar.button("🔁 Reset Filters"):
    st.experimental_rerun()

# Clean & Process
filtered_data['doctor_available'] = filtered_data['doctor_available'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
filtered_data['waiting_time_minutes'] = pd.to_numeric(filtered_data['waiting_time_minutes'], errors='coerce')
filtered_data.dropna(subset=['waiting_time_minutes'], inplace=True)

# VISUALS

#Waiting Time Distribution
st.subheader("Waiting Time Distribution")
waiting_time_summary = filtered_data.groupby('time_block')['waiting_time_minutes'].mean().reset_index()
fig = px.bar(waiting_time_summary, x='time_block', y='waiting_time_minutes', title="Average Waiting Time by Time Block")
st.plotly_chart(fig, use_container_width=True)

if not filtered_data.empty:
    avg_waiting_time = filtered_data['waiting_time_minutes'].mean()
    st.write(f"**Predicted Avg. Waiting Time:** {avg_waiting_time:.2f} minutes")
else:
    st.write("**Predicted Avg. Waiting Time:** No data available.")

st.subheader("Congestion Level")
fig = px.histogram(filtered_data, x='congestion_level', title="Congestion Level Distribution")
st.plotly_chart(fig, use_container_width=True)
mode_cong = filtered_data['congestion_level'].mode()[0]
predicted_cong = mode_cong if mode_cong in label_encoder.classes_ else "Unknown"
st.write(f"**Predicted Congestion Level:** {predicted_cong}")

st.subheader("Expected Walk-Ins")
fig = px.bar(filtered_data, x='time_block', y='expected_walk_ins', color='day_of_week', title="Expected Walk-Ins by Time")
st.plotly_chart(fig, use_container_width=True)
st.write(f"**Total Expected Walk-Ins:** {filtered_data['expected_walk_ins'].sum()} patients")

st.subheader("Doctor Availability")
fig = px.bar(filtered_data, x='time_block', y='doctor_available', color='day_of_week', title="Doctor Availability by Time Block")
st.plotly_chart(fig, use_container_width=True)
doc_mean = filtered_data['doctor_available'].mean()
st.write(f"**Doctor Status:** {'Available' if doc_mean > 0 else 'Not Available'}")

st.subheader("Patient Load Ratio")
load_ratio_summary = filtered_data.groupby('time_block')['patient_load_ratio'].mean().reset_index()
fig = px.bar(load_ratio_summary, x='time_block', y='patient_load_ratio', title="Patient Load Ratio by Time Block")
st.plotly_chart(fig, use_container_width=True)

if not filtered_data.empty:
    avg_load_ratio = filtered_data['patient_load_ratio'].mean()
    st.write(f"**Avg. Load Ratio:** {avg_load_ratio:.2f} patients per doctor")
else:
    st.write("**Avg. Load Ratio:** No data available.")

st.subheader("Doctor to Patient Ratio")
doc_patient_summary = filtered_data.groupby('time_block')['doctor_patient_ratio'].mean().reset_index()
fig = px.bar(doc_patient_summary, x='time_block', y='doctor_patient_ratio', title="Doctor to Patient Ratio by Time Block")
st.plotly_chart(fig, use_container_width=True)

if not filtered_data.empty:
    avg_doc_patient_ratio = filtered_data['doctor_patient_ratio'].mean()
    st.write(f"**Avg. Doc/Patient Ratio:** {avg_doc_patient_ratio:.2f}")
else:
    st.write("**Avg. Doc/Patient Ratio:** No data available.")

st.subheader("Emergencies & Seasonal Illnesses")
fig = px.line(filtered_data, x='time_block', y=['emergencies', 'seasonal_illnesses'], markers=True)
st.plotly_chart(fig, use_container_width=True)
st.write(f"**Emergencies:** {filtered_data['emergencies'].sum()}")
st.write(f"**Seasonal Illnesses:** {filtered_data['seasonal_illnesses'].sum()}")

st.subheader("Weather Impact")
fig = px.scatter(filtered_data, x='temperature', y='actual_patients', color='humidity')
st.plotly_chart(fig, use_container_width=True)

if not filtered_data.empty:
    avg_temp = filtered_data['temperature'].mean()
    avg_humidity = filtered_data['humidity'].mean()
    st.write(f"**Avg. Temperature:** {avg_temp:.2f}°C")
    st.write(f"**Avg. Humidity:** {avg_humidity:.2f}%")
else:
    st.write("**Avg. Temperature:** No data available.")
    st.write("**Avg. Humidity:** No data available.")

# Input Form
st.subheader('Please Input Hospital Records')
with st.form("input_form"):
    input_hospital = st.text_input("Hospital Name", ['Mbagathi', 'KNH', 'Mama Lucy', 'Pumwani', 'Kenyatta'])
    input_department = st.text_input("Department", ['Pediatrics', 'Emergency', 'Maternity', 'General', 'Surgery'])
    input_date = st.date_input("Date")
    input_waiting_time = st.number_input("Waiting Time (minutes)", min_value=0)
    input_congestion_level = st.selectbox("Congestion Level", ["Low", "Medium", "High"])
    input_walk_ins = st.number_input("Expected Walk-Ins", min_value=0)
    input_doctor_available = st.selectbox("Doctor Available", ["Yes", "No"])
    input_patient_load_ratio = st.number_input("Patient Load Ratio", min_value=0.0)
    input_doctor_patient_ratio = st.number_input("Doctor Patient Ratio", min_value=0.0)
    input_emergencies = st.number_input("Emergencies", min_value=0)
    input_seasonal_illnesses = st.number_input("Seasonal Illnesses", min_value=0)
    input_public_holidays = st.number_input("Public Holidays/Events", min_value=0)
    input_temperature = st.number_input("Temperature (°C)", min_value=0.0)
    input_humidity = st.number_input("Humidity (%)", min_value=0.0)

    submitted = st.form_submit_button("Submit")
    if submitted:
        input_data = pd.DataFrame({
            'hospital_name': [input_hospital],
            'department': [input_department],
            'date': [input_date],
            'waiting_time_minutes': [input_waiting_time],
            'congestion_level': [input_congestion_level],
            'expected_walk_ins': [input_walk_ins],
            'doctor_available': [input_doctor_available],
            'patient_load_ratio': [input_patient_load_ratio],
            'doctor_patient_ratio': [input_doctor_patient_ratio],
            'emergencies': [input_emergencies],
            'seasonal_illnesses': [input_seasonal_illnesses],
            'public_holidays_events': [input_public_holidays],
            'temperature': [input_temperature],
            'humidity': [input_humidity]
        })
        st.success("Record captured below:")
        st.dataframe(input_data)


st.markdown("### Tooltips")
st.write("**Waiting Time Distribution:** Shows the distribution of waiting times for patients.")
st.write("**Congestion Level Distribution:** Shows the distribution of congestion levels at the hospital.")
st.write("**Expected Walk-In Traffic:** Shows the expected walk-in traffic for different time blocks.")
st.write("**Doctor Availability:** Shows the availability of doctors for different time blocks.")
st.write("**Patient Load Ratio:** Shows the ratio of patients to doctors.")
st.write("**Doctor Patient Ratio:** Shows the ratio of doctors to patients.")
st.write("**Emergencies and Seasonal Illnesses:** Shows the trends in emergencies and seasonal diseases.")
