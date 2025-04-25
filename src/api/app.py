from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

data = request.get_json(force=True)
features = np.array(data['features'])

print(f"Shape of features: {features.shape}")

# Load the trained models and preprocessor
rf_regressor = joblib.load('/home/user/Desktop/Wapi Daktari/src/api/random_forest_regressor.pkl')
xgb_regressor = joblib.load('/home/user/Desktop/Wapi Daktari/src/api/xgboost_regressor.pkl')
hybrid_regressor = joblib.load('/home/user/Desktop/Wapi Daktari/src/api/hybrid_regressor.pkl')
rf_classifier = joblib.load('/home/user/Desktop/Wapi Daktari/src/api/random_forest_classifier.pkl')
xgb_classifier = joblib.load('/home/user/Desktop/Wapi Daktari/src/api/xgboost_classifier.pkl')
hybrid_classifier = joblib.load('/home/user/Desktop/Wapi Daktari/src/api/hybrid_classifier.pkl')
preprocessor = joblib.load('/home/user/Desktop/Wapi Daktari/src/api/preprocessor.pkl')
label_encoder = joblib.load('/home/user/Desktop/Wapi Daktari/src/api/label_encoder.pkl')

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


if __name__ == '__main__':
    app.run(debug=True)