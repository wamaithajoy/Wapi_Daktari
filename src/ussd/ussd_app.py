import traceback
from flask import Flask, request
import joblib
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging
import os
app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and preprocessor
base_path = os.path.dirname(os.path.abspath(__file__))
rf_regressor = joblib.load(os.path.join(base_path, '..', 'api', 'random_forest_regressor.pkl'))
xgb_regressor = joblib.load(os.path.join(base_path, '..', 'api', 'xgboost_regressor.pkl'))
hybrid_regressor = joblib.load(os.path.join(base_path, '..', 'api', 'hybrid_regressor.pkl'))
rf_classifier = joblib.load(os.path.join(base_path, '..', 'api', 'random_forest_classifier.pkl'))
xgb_classifier = joblib.load(os.path.join(base_path, '..', 'api', 'xgboost_classifier.pkl'))
hybrid_classifier = joblib.load(os.path.join(base_path, '..', 'api', 'hybrid_classifier.pkl'))
preprocessor = joblib.load(os.path.join(base_path, '..', 'api', 'preprocessor.pkl'))
label_encoder = joblib.load(os.path.join(base_path, '..', 'api', 'label_encoder.pkl'))

# Load the dataset
df = pd.read_csv(os.path.join(base_path, '..', '..', 'data', 'wapi_daktari_healthcare_dataset.csv'))

# Define dropdown values
HOSPITALS = df['hospital_name'].unique().tolist()
DEPARTMENTS = df['department'].unique().tolist()

# Define time blocks to evaluate
TIMEBLOCKS = {
    "Morning": {"hour_of_day": 8},
    "Afternoon": {"hour_of_day": 15},
    "Evening": {"hour_of_day": 18},
}

TRANSLATIONS = {
    'en': {
        'welcome': "Welcome to Wapi Daktari",
        'language_select': "Select language:\n1. English\n2. Kiswahili",
        'main_menu': "1. Check best time to visit hospital\n2. Change language",
        'select_hospital': "Select Hospital:",
        'select_department': "Select Department:",
        'select_date': "Select Date:",
        'enter_date': "Enter the date (YYYY-MM-DD):",
        'back': "0. Back",
        'invalid_input': "Invalid input. Try again.",
        'error_occurred': "An error occurred:",
        'best_time': "Best time to visit",
        'time': "Time:",
        'estimated_waiting_time': "Estimated Waiting Time:",
        'expected_congestion': "Expected Congestion:",
        'minutes': "minutes",
        'today': "Today",
        'tomorrow': "Tomorrow",
        'day_after_tomorrow': "Day after tomorrow",
        'enter_specific_date': "Enter specific date",
    },
    'sw': {
        'welcome': "Karibu Wapi Daktari",
        'language_select': "Chagua lugha:\n1. Kiingereza\n2. Kiswahili",
        'main_menu': "1. Angalia wakati bora wa kutembelea hospitali\n2. Badilisha lugha",
        'select_hospital': "Chagua Hospitali:",
        'select_department': "Chagua Idara:",
        'select_date': "Chagua Tarehe:",
        'enter_date': "Ingiza tarehe (YYYY-MM-DD):",
        'back': "0. Rudi nyuma",
        'invalid_input': "Ingizo batili. Jaribu tena.",
        'error_occurred': "Kosa limetokea:",
        'best_time': "Wakati bora wa kutembelea",
        'time': "Wakati:",
        'estimated_waiting_time': "Muda wa Kusubiri Unakadiriwa:",
        'expected_congestion': "Msongamano Unatarajiwa:",
        'minutes': "dakika",
        'today': "Leo",
        'tomorrow': "Kesho",
        'day_after_tomorrow': "Kesho kutwa",
        'enter_specific_date': "Ingiza tarehe maalum",
    }
}

def get_features(hospital_name, department, date_obj, time_block):
    # Filter the dataset for the given hospital, department, and date
    filtered_df = df[(df['hospital_name'] == hospital_name) & 
                     (df['department'] == department) & 
                     (df['date'] == date_obj.strftime('%Y-%m-%d')) &
                     (df['time_block'] == time_block)]
    
    if filtered_df.empty:
        raise ValueError(f"No data available for {hospital_name}, {department} on {date_obj.strftime('%Y-%m-%d')} during {time_block}")
    
    # If multiple rows are found, use the first one
    features = filtered_df.iloc[0].to_dict()
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    return features_df

def predict_best_time(hospital_name, department, date_obj):
    best_time = None
    min_waiting_time = float('inf')
    best_congestion = None

    for time_block in TIMEBLOCKS.keys():
        try:
            features = get_features(hospital_name, department, date_obj, time_block)
            
            print(f"Debug: Raw features for {time_block}:")
            print(features)
            
            # Preprocess features
            X = preprocessor.transform(features)
            print(f"Debug: Preprocessed features shape for {time_block}: {X.shape}")
            
            # Make predictions
            waiting_time = rf_regressor.predict(X)[0]
            congestion = rf_classifier.predict(X)[0]
            print(f"Debug: Predictions for {time_block}: Waiting time = {waiting_time}, Congestion = {congestion}")
            
            if waiting_time < min_waiting_time:
                min_waiting_time = waiting_time
                best_time = time_block
                best_congestion = congestion
        
        except Exception as e:
            print(f"Error for {time_block}: {str(e)}")
            continue

    if best_time is None:
        raise ValueError("Unable to make predictions for any time block")

    # Handle the case where the label is not in the encoder
    try:
        congestion_label = label_encoder.inverse_transform([best_congestion])[0]
    except ValueError:
        congestion_label = str(best_congestion)  # Use the raw prediction if it can't be inverse transformed

    return best_time, min_waiting_time, congestion_label

@app.route("/ussd", methods=["POST"])
def ussd():
    session_id = request.form.get("sessionId")
    phone_number = request.form.get("phoneNumber")
    text = request.form.get("text")
    
    logger.info(f"Received USSD request: SessionID: {session_id}, Phone: {phone_number}, Text: {text}")

    inputs = text.split("*")
    step = len(inputs)

    # Default language is English
    lang = 'en'

    if text == "":
        return menu(TRANSLATIONS[lang]['language_select'])

    if step == 1:
        lang = 'en' if inputs[0] == '1' else 'sw'
        return menu(f"{TRANSLATIONS[lang]['welcome']}\n{TRANSLATIONS[lang]['main_menu']}")

    lang = 'en' if inputs[0] == '1' else 'sw'

    if inputs[1] == "1":
        if step == 2:
            hospital_menu = "\n".join([f"{i+1}. {name}" for i, name in enumerate(HOSPITALS)])
            return menu(f"{TRANSLATIONS[lang]['select_hospital']}\n{hospital_menu}\n\n{TRANSLATIONS[lang]['back']}")
        elif step == 3:
            if inputs[2] == "0":
                return menu(f"{TRANSLATIONS[lang]['welcome']}\n{TRANSLATIONS[lang]['main_menu']}")
            department_menu = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS)])
            return menu(f"{TRANSLATIONS[lang]['select_department']}\n{department_menu}\n\n{TRANSLATIONS[lang]['back']}")
        elif step == 4:
            if inputs[3] == "0":
                hospital_menu = "\n".join([f"{i+1}. {name}" for i, name in enumerate(HOSPITALS)])
                return menu(f"{TRANSLATIONS[lang]['select_hospital']}\n{hospital_menu}\n\n{TRANSLATIONS[lang]['back']}")
            today = datetime.now()
            date_menu = (
                f"1. {TRANSLATIONS[lang]['today']} ({today.strftime('%Y-%m-%d')})\n"
                f"2. {TRANSLATIONS[lang]['tomorrow']} ({(today + timedelta(days=1)).strftime('%Y-%m-%d')})\n"
                f"3. {TRANSLATIONS[lang]['day_after_tomorrow']} ({(today + timedelta(days=2)).strftime('%Y-%m-%d')})\n"
                f"4. {TRANSLATIONS[lang]['enter_specific_date']}"
            )
            return menu(f"{TRANSLATIONS[lang]['select_date']}\n{date_menu}\n\n{TRANSLATIONS[lang]['back']}")
        # ... rest of the function ...
        elif step == 5:
            if inputs[4] == "0":
                department_menu = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS)])
                return menu(f"{TRANSLATIONS[lang]['select_department']}\n{department_menu}\n\n{TRANSLATIONS[lang]['back']}")
            elif inputs[4] == "4":
                return menu(f"{TRANSLATIONS[lang]['enter_date']}\n\n{TRANSLATIONS[lang]['back']}")
            else:
                try:
                    hospital_name = HOSPITALS[int(inputs[2]) - 1]
                    department = DEPARTMENTS[int(inputs[3]) - 1]
                    date_choice = int(inputs[4])
                    if date_choice == 1:
                        date_obj = datetime.now()
                    elif date_choice == 2:
                        date_obj = datetime.now() + timedelta(days=1)
                    elif date_choice == 3:
                        date_obj = datetime.now() + timedelta(days=2)
                    else:
                        return end(TRANSLATIONS[lang]['invalid_input'])

                    best_time, waiting_time, congestion = predict_best_time(hospital_name, department, date_obj)

                    response = f"{TRANSLATIONS[lang]['best_time']} {hospital_name} - {department} on {date_obj.strftime('%Y-%m-%d')}:\n"
                    response += f"{TRANSLATIONS[lang]['time']} {best_time}\n"
                    response += f"{TRANSLATIONS[lang]['estimated_waiting_time']} {waiting_time:.0f} {TRANSLATIONS[lang]['minutes']}\n"
                    response += f"{TRANSLATIONS[lang]['expected_congestion']} {congestion}"

                    return end(response)
                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"Error occurred: {str(e)}\n{error_trace}")
                    return end(f"{TRANSLATIONS[lang]['error_occurred']} {str(e)}")
        elif step == 6:
            if inputs[5] == "0":
                today = datetime.now()
                date_menu = (
                    f"1. {TRANSLATIONS[lang]['today']} ({today.strftime('%Y-%m-%d')})\n"
                    f"2. {TRANSLATIONS[lang]['tomorrow']} ({(today + timedelta(days=1)).strftime('%Y-%m-%d')})\n"
                    f"3. {TRANSLATIONS[lang]['day_after_tomorrow']} ({(today + timedelta(days=2)).strftime('%Y-%m-%d')})\n"
                    f"4. {TRANSLATIONS[lang]['enter_specific_date']}"
                )
                return menu(f"{TRANSLATIONS[lang]['select_date']}\n{date_menu}\n\n{TRANSLATIONS[lang]['back']}")
            try:
                hospital_name = HOSPITALS[int(inputs[2]) - 1]
                department = DEPARTMENTS[int(inputs[3]) - 1]
                date_obj = datetime.strptime(inputs[5], "%Y-%m-%d")

                best_time, waiting_time, congestion = predict_best_time(hospital_name, department, date_obj)

                response = f"{TRANSLATIONS[lang]['best_time']} {hospital_name} - {department} on {date_obj.strftime('%Y-%m-%d')}:\n"
                response += f"{TRANSLATIONS[lang]['time']} {best_time}\n"
                response += f"{TRANSLATIONS[lang]['estimated_waiting_time']} {waiting_time:.0f} {TRANSLATIONS[lang]['minutes']}\n"
                response += f"{TRANSLATIONS[lang]['expected_congestion']} {congestion}"

                return end(response)
            except Exception as e:
                error_trace = traceback.format_exc()
                print(f"Error occurred: {str(e)}\n{error_trace}")
                return end(f"{TRANSLATIONS[lang]['error_occurred']} {str(e)}")
    elif inputs[1] == "2":
        # Change language
        new_lang = 'sw' if lang == 'en' else 'en'
        return menu(f"{TRANSLATIONS[new_lang]['welcome']}\n{TRANSLATIONS[new_lang]['main_menu']}")

    return end(TRANSLATIONS[lang]['invalid_input'])

def menu(response):
    return f"CON {response}"

def end(response):
    return f"END {response}"