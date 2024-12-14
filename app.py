
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('models/scaler_clustering.pkl', 'rb') as f:
    scaler_clustering = pickle.load(f)

with open('models/rf_model_with_gmm.pkl', 'rb') as f:
    rf_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data
        age = float(request.form['age'])
        net_sales = float(request.form['net_sales'])
        commission = float(request.form['commission'])
        duration = float(request.form['duration'])

        # Clustering
        input_data = np.array([[age, net_sales, commission, duration]])
        scaled_data = scaler_clustering.transform(input_data)
        cluster_label = kmeans_model.predict(scaled_data)[0]

        # Supervised prediction
        input_data_with_cluster = np.array([[age, net_sales, commission, duration, cluster_label]])
        rf_prediction = rf_model.predict(input_data_with_cluster)[0]
        rf_prediction_proba = rf_model.predict_proba(input_data_with_cluster)[0]

        result_label = 'Yes' if rf_prediction == 1 else 'No'
        probability = rf_prediction_proba[1] if rf_prediction == 1 else rf_prediction_proba[0]

        return render_template(
            'result.html',
            age=age,
            net_sales=net_sales,
            commission=commission,
            duration=duration,
            cluster_label=cluster_label,
            result=result_label,
            probability=f"{probability:.2f}"
        )
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get('PORT', 5000)))
