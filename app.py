from flask import Flask, request,jsonify
import joblib
import numpy as np

app = flask(__name__)

#load model
model = joblib.load('regression_model.pkl')
@app.route('/predict', method=['POST'])
def predict():
    data = request.get_json()  # Ambil data dari request JSON
    X_new = np.array(data['features']).reshape(1, -1)  # Konversi ke array numpy
    prediction = model.predict(X_new)[0]  # Prediksi
    return jsonify({'predicted_sales': prediction})  # Hasil dalam format JSON

if __name__ == '__main__':
    app.run(debug=True)