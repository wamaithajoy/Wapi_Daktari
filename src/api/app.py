from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained models and preprocessor
base_path = os.path.dirname(os.path.abspath(__file__))
rf_regressor = joblib.load(os.path.join(base_path, 'random_forest_regressor.pkl'))
xgb_regressor = joblib.load(os.path.join(base_path, 'xgboost_regressor.pkl'))
hybrid_regressor = joblib.load(os.path.join(base_path, 'hybrid_regressor.pkl'))
rf_classifier = joblib.load(os.path.join(base_path, 'random_forest_classifier.pkl'))
xgb_classifier = joblib.load(os.path.join(base_path, 'xgboost_classifier.pkl'))
hybrid_classifier = joblib.load(os.path.join(base_path, 'hybrid_classifier.pkl'))
preprocessor = joblib.load(os.path.join(base_path, 'preprocessor.pkl'))
label_encoder = joblib.load(os.path.join(base_path, 'label_encoder.pkl'))

@app.route('/')
def home():
    return "Welcome to Wapi Daktari API"

@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    data = request.get_json(force=True)
    features = preprocessor.transform(np.array(data['features']).reshape(1, -1))
    prediction_rf = rf_regressor.predict(features)
    prediction_xgb = xgb_regressor.predict(features)
    prediction_hybrid = hybrid_regressor.predict(features)
    return jsonify({
        'random_forest': prediction_rf.tolist(),
        'xgboost': prediction_xgb.tolist(),
        'hybrid': prediction_hybrid.tolist()
    })

@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    data = request.get_json(force=True)
    features = preprocessor.transform(np.array(data['features']).reshape(1, -1))
    prediction_rf = rf_classifier.predict(features)
    prediction_xgb = xgb_classifier.predict(features)
    prediction_hybrid = hybrid_classifier.predict(features)
    return jsonify({
        'random_forest': label_encoder.inverse_transform(prediction_rf).tolist(),
        'xgboost': label_encoder.inverse_transform(prediction_xgb).tolist(),
        'hybrid': label_encoder.inverse_transform(prediction_hybrid).tolist()
    })

@app.route('/ussd', methods=['POST'])
def ussd_callback():
    session_id = request.values.get("sessionId", None)
    service_code = request.values.get("serviceCode", None)
    phone_number = request.values.get("phoneNumber", None)
    text = request.values.get("text", "")

    user_response = text.strip().split('*')
    level = len(user_response)

    if text == "":
        response = "CON Welcome to Wapi Daktari\n"
        response += "1. Predict Regression\n"
        response += "2. Predict Classification"
    elif text == "1":
        response = "CON Enter features (comma-separated):"
    elif text == "2":
        response = "CON Enter features (comma-separated):"
    elif user_response[0] == "1" and level == 2:
        try:
            features = [float(i) for i in user_response[1].split(',')]
            features = preprocessor.transform(np.array(features).reshape(1, -1))
            prediction_rf = rf_regressor.predict(features)[0]
            prediction_xgb = xgb_regressor.predict(features)[0]
            prediction_hybrid = hybrid_regressor.predict(features)[0]
            response = f"END RF: {prediction_rf:.2f}, XGB: {prediction_xgb:.2f}, Hybrid: {prediction_hybrid:.2f}"
        except:
            response = "END Invalid input. Use comma-separated numbers."
    elif user_response[0] == "2" and level == 2:
        try:
            features = [float(i) for i in user_response[1].split(',')]
            features = preprocessor.transform(np.array(features).reshape(1, -1))
            prediction_rf = label_encoder.inverse_transform(rf_classifier.predict(features))[0]
            prediction_xgb = label_encoder.inverse_transform(xgb_classifier.predict(features))[0]
            prediction_hybrid = label_encoder.inverse_transform(hybrid_classifier.predict(features))[0]
            response = f"END RF: {prediction_rf}, XGB: {prediction_xgb}, Hybrid: {prediction_hybrid}"
        except:
            response = "END Invalid input. Use comma-separated numbers."
    else:
        response = "END Invalid choice. Try again."

    return response

if __name__ == '__main__':
    if os.environ.get('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    else:
        app.run(debug=True)
