# data/generate_dataset.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define constants
HOSPITALS = ['Mbagathi', 'KNH', 'Mama Lucy', 'Pumwani', 'Kenyatta']
DEPARTMENTS = ['Pediatrics', 'Emergency', 'Maternity', 'General', 'Surgery']
TIME_BLOCKS = ['Morning', 'Afternoon', 'Evening']
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)  # Extended to cover a full year

# Helper function to generate dates
def generate_dates(start_date, end_date):
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

# Helper function to determine if a date is a weekend
def is_weekend(date):
    return date.weekday() >= 5

# Helper function to determine if a date is a holiday
def is_holiday(date):
    holidays = [
        datetime(2025, 1, 1),  # New Year's Day
        datetime(2025, 4, 18),  # Easter
        datetime(2025, 5, 1),  # Labor Day
        datetime(2025, 6, 1),  # Madaraka Day
        datetime(2025, 10, 20),  # Mashujaa Day
        datetime(2025, 12, 12),  # Jamhuri Day
        datetime(2025, 12, 25),  # Christmas Day
        datetime(2025, 12, 26)   # Boxing Day
    ]
    return date in holidays

# Helper function to determine if a date is a strike day
def is_strike_day(date):
    strike_days = [
        datetime(2025, 2, 15),  # Example strike day
        datetime(2025, 3, 20),  # Example strike day
        datetime(2025, 11, 5)   # Example strike day
    ]
    return date in strike_days

# Helper function to simulate doctor availability
def simulate_doctor_availability(date, is_holiday, is_strike_day, hospital):
    if is_holiday or is_strike_day:
        return 0, 'No'
    if hospital == 'Mbagathi' or hospital == 'Pumwani':  # Smaller hospitals
        doctors_on_shift = random.randint(0, 3)
    else:  # Larger hospitals
        doctors_on_shift = random.randint(0, 5)
    doctor_available = 'Yes' if doctors_on_shift > 0 else 'No'
    return doctors_on_shift, doctor_available

# Helper function to simulate patient traffic
def simulate_patient_traffic(time_block, is_weekend, is_holiday, department, hospital):
    base_patients = 50
    if time_block == 'Morning':
        base_patients += 20
    if is_weekend:
        base_patients -= 10
    if is_holiday:
        base_patients += 15
    if department == 'Emergency':
        base_patients += 10
    if hospital == 'KNH' or hospital == 'Kenyatta':  # Larger hospitals
        base_patients += 20
    return base_patients + random.randint(-5, 5)

# Helper function to simulate waiting time
def simulate_waiting_time(doctors_on_shift, actual_patients):
    if doctors_on_shift == 0:
        return random.randint(90, 120)
    else:
        return random.randint(30, 60)

# Helper function to simulate doctor arrival delay
def simulate_doctor_arrival_delay():
    return random.randint(0, 120)

# Helper function to determine congestion level
def determine_congestion_level(actual_patients):
    if actual_patients < 50:
        return 'Low'
    elif actual_patients < 80:
        return 'Medium'
    else:
        return 'High'

# Helper function to simulate walk-in traffic
def simulate_walk_in_traffic(time_block, is_weekend, is_holiday, month, hospital):
    base_walk_ins = 50
    if time_block == 'Morning':
        base_walk_ins += 20
    if is_weekend:
        base_walk_ins -= 10
    if is_holiday:
        base_walk_ins += 15
    if month in [1, 2, 12]:  # Flu season
        base_walk_ins += 10
    if hospital == 'KNH' or hospital == 'Kenyatta':  # Larger hospitals
        base_walk_ins += 20
    return base_walk_ins + random.randint(-5, 5)

# Helper function to simulate emergencies
def simulate_emergencies(month):
    if month in [6, 7, 8]:  # Malaria season
        return random.randint(5, 15)
    else:
        return random.randint(0, 5)

# Helper function to simulate seasonal illnesses
def simulate_seasonal_illnesses(month):
    if month in [1, 2, 12]:  # Flu season
        return random.randint(5, 15)
    else:
        return random.randint(0, 5)

# Helper function to simulate public holidays and events
def simulate_public_holidays_events(date):
    if is_holiday(date):
        return random.randint(-20, -10)
    else:
        return 0

# Generate dates
dates = generate_dates(START_DATE, END_DATE)

# Initialize the DataFrame
data = []

# Generate data for each date
for date in dates:
    for hospital in HOSPITALS:
        for department in DEPARTMENTS:
            for time_block in TIME_BLOCKS:
                # Determine day of the week, weekend, holiday, and strike day
                day_of_week = date.weekday()
                is_weekend_flag = is_weekend(date)
                is_holiday_flag = is_holiday(date)
                is_strike_day_flag = is_strike_day(date)

                # Simulate doctor availability
                doctors_on_shift, doctor_available = simulate_doctor_availability(date, is_holiday_flag, is_strike_day_flag, hospital)

                # Simulate patient traffic
                expected_patients = simulate_patient_traffic(time_block, is_weekend_flag, is_holiday_flag, department, hospital)
                actual_patients = expected_patients + random.randint(-5, 5)

                # Simulate waiting time
                waiting_time_minutes = simulate_waiting_time(doctors_on_shift, actual_patients)

                # Simulate peak hour
                peak_hour = random.choice(['Yes', 'No'])

                # Simulate doctor arrival delay
                doctor_arrival_delay = simulate_doctor_arrival_delay()

                # Determine congestion level
                congestion_level = determine_congestion_level(actual_patients)

                # Simulate walk-in traffic
                expected_walk_ins = simulate_walk_in_traffic(time_block, is_weekend_flag, is_holiday_flag, date.month, hospital)

                # Simulate emergencies
                emergencies = simulate_emergencies(date.month)

                # Simulate seasonal illnesses
                seasonal_illnesses = simulate_seasonal_illnesses(date.month)

                # Simulate public holidays and events
                public_holidays_events = simulate_public_holidays_events(date)

                # Simulate additional time-based features
                hour_of_day = date.hour
                day_of_month = date.day
                quarter = (date.month - 1) // 3 + 1
                season = 'Spring' if date.month in [3, 4, 5] else 'Summer' if date.month in [6, 7, 8] else 'Autumn' if date.month in [9, 10, 11] else 'Winter'

                # Simulate lag features
                previous_day_patients = actual_patients - random.randint(0, 5)
                previous_week_patients = actual_patients - random.randint(0, 10)
                previous_month_patients = actual_patients - random.randint(0, 20)
                previous_day_waiting_time = waiting_time_minutes - random.randint(0, 10)
                previous_week_waiting_time = waiting_time_minutes - random.randint(0, 20)
                previous_month_waiting_time = waiting_time_minutes - random.randint(0, 30)

                # Simulate interaction features
                doctor_patient_ratio = doctors_on_shift / actual_patients
                doctors_on_shift_expected_patients = doctors_on_shift * expected_patients
                
                cong_num = {'Low': 0, 'Medium': 1, 'High': 2}[congestion_level]
                doctor_patient_ratio_congestion_level = doctor_patient_ratio * cong_num

                # Simulate seasonal features
                current_month = date.month
                flu_season    = 1 if current_month in [1, 2, 3] else 0
                malaria_season= 1 if current_month in [6, 7, 8] else 0
                
                mon = date.month
                if mon in [12,1,2]:    # Summer in Nairobi
                    temperature = random.uniform(25.0, 35.0)
                    humidity    = random.uniform(40.0, 80.0)
                    rainfall    = random.uniform(0.0, 10.0)
                elif mon in [3,4,5]:   # Long rains
                    temperature = random.uniform(20.0, 30.0)
                    humidity    = random.uniform(50.0, 90.0)
                    rainfall    = random.uniform(5.0, 30.0)
                elif mon in [6,7,8]:   # Cool dry
                    temperature = random.uniform(15.0, 25.0)
                    humidity    = random.uniform(30.0, 70.0)
                    rainfall    = random.uniform(0.0, 5.0)
                else:                  # Short rains
                    temperature = random.uniform(18.0, 28.0)
                    humidity    = random.uniform(45.0, 85.0)
                    rainfall    = random.uniform(2.0, 25.0)

                # Simulate holiday and event features
                school_holidays = random.choice([0, 1])
                national_events = random.choice([0, 1])

                # Simulate historical trends
                average_waiting_time_last_week = waiting_time_minutes - random.randint(0, 10)
                average_patients_last_month = actual_patients - random.randint(0, 20)

                # Append to data list
                data.append([
                    hospital, date, day_of_week, is_weekend_flag, is_holiday_flag,
                    is_strike_day_flag, department, time_block, doctors_on_shift,
                    expected_patients, actual_patients, waiting_time_minutes,
                    peak_hour, doctor_available, doctor_arrival_delay, congestion_level,
                    date.month, date.day, actual_patients / expected_patients,
                    doctors_on_shift / actual_patients, is_holiday_flag * is_strike_day_flag,
                    expected_walk_ins, emergencies, seasonal_illnesses, public_holidays_events,
                    hour_of_day, day_of_month, quarter, season, previous_day_patients,
                    previous_week_patients, previous_month_patients, temperature, humidity, rainfall,
                    school_holidays, national_events, average_waiting_time_last_week, average_patients_last_month,
                    previous_day_waiting_time, previous_week_waiting_time, previous_month_waiting_time,
                    doctors_on_shift_expected_patients, doctor_patient_ratio_congestion_level,
                    flu_season, malaria_season
                ])

# Convert to DataFrame
columns = [
    'hospital_name', 'date', 'day_of_week', 'is_weekend', 'is_holiday',
    'is_strike_day', 'department', 'time_block', 'doctors_on_shift',
    'expected_patients', 'actual_patients', 'waiting_time_minutes',
    'peak_hour', 'doctor_available', 'doctor_arrival_delay', 'congestion_level',
    'month', 'day', 'patient_load_ratio', 'doctor_patient_ratio',
    'holiday_strike_interaction', 'expected_walk_ins', 'emergencies',
    'seasonal_illnesses', 'public_holidays_events', 'hour_of_day', 'day_of_month',
    'quarter', 'season', 'previous_day_patients', 'previous_week_patients',
    'previous_month_patients', 'temperature', 'humidity', 'rainfall',
    'school_holidays', 'national_events', 'average_waiting_time_last_week', 'average_patients_last_month',
    'previous_day_waiting_time', 'previous_week_waiting_time', 'previous_month_waiting_time',
    'doctors_on_shift_expected_patients', 'doctor_patient_ratio_congestion_level',
    'flu_season', 'malaria_season'
]
df = pd.DataFrame(data, columns=columns)

df.to_csv('../data/wapi_daktari_healthcare_dataset.csv', index=False)
