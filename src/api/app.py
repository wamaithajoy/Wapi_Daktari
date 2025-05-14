from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os
import logging

logging.basicConfig(level=logging.DEBUG)
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
    app.logger.debug("Home route accessed")
    return "Welcome to Wapi Daktari API. Visit /docs for API documentation."

@app.route('/test')
def test():
    return "Test endpoint working"

@app.route('/predict_regression', methods=['GET', 'POST'])
def predict_regression():
    if request.method == 'GET':
        return "This is the predict_regression endpoint. Use POST method with JSON data to make predictions."
    
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

@app.route('/predict_classification', methods=['GET', 'POST'])
def predict_classification():
    if request.method == 'GET':
        return "This is the predict_classification endpoint. Use POST method with JSON data to make predictions."
    
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

@app.route('/ussd', methods=['GET', 'POST'])
def ussd_callback():
    if request.method == 'GET':
        return "This is the USSD endpoint. Use POST method to interact with the USSD service."
    
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

@app.route('/docs')
def docs():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Wapi Daktari API Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
            h2 { color: #666; }
            pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; }
            .endpoint { margin-bottom: 30px; }
        </style>
    </head>
    <body>
        <h1>Wapi Daktari API Documentation</h1>
        
        <div class="endpoint">
            <h2>1. Predict Regression</h2>
            <p><strong>Endpoint:</strong> /predict_regression</p>
            <p><strong>Method:</strong> POST</p>
            <p><strong>Description:</strong> Predicts waiting time using regression models.</p>
            <p><strong>Example Request:</strong></p>
            <pre>
curl -X POST https://wapi-daktari.onrender.com/predict_regression \
-H "Content-Type: application/json" \
-d '{"features": [1, 2, 3, 4, 5]}'
            </pre>
        </div>

        <div class="endpoint">
            <h2>2. Predict Classification</h2>
            <p><strong>Endpoint:</strong> /predict_classification</p>
            <p><strong>Method:</strong> POST</p>
            <p><strong>Description:</strong> Predicts congestion level using classification models.</p>
            <p><strong>Example Request:</strong></p>
            <pre>
curl -X POST https://wapi-daktari.onrender.com/predict_classification \
-H "Content-Type: application/json" \
-d '{"features": [1, 2, 3, 4, 5]}'
            </pre>
        </div>

        <div class="endpoint">
            <h2>3. USSD Service</h2>
            <p><strong>Endpoint:</strong> /ussd</p>
            <p><strong>Method:</strong> POST</p>
            <p><strong>Description:</strong> Handles USSD interactions for the service.</p>
            <p><strong>Note:</strong> This endpoint is typically accessed through a USSD gateway and not directly by users.</p>
        </div>
        
          <div class="endpoint">
            <h2>4. Wapi Daktari Dashboard</h2>
            <p><strong>URL:</strong> <a href="https://wapidaktari-lwhs69lmrfyyfd7cnqww9o.streamlit.app/" target="_blank">https://wapidaktari-lwhs69lmrfyyfd7cnqww9o.streamlit.app/</a></p>
            <p><strong>Description:</strong> Interactive dashboard for visualizing Wapi Daktari data and predictions.</p>
        </div>

        <p>For more information or support, please contact the development team.</p>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    if os.environ.get('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    else:
        app.run(debug=True)