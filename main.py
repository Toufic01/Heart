import os
import pickle as pk
from flask import Flask, request, jsonify
import numpy as np
from flask_sslify import SSLify

app = Flask(__name__)
sslify = SSLify(app)

# Use a relative path for the model file
model_path = 'models/model'

# Check if the file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

with open(model_path, 'rb') as model_file:
    model = pk.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sex = float(request.form.get('sex'))
        age = float(request.form.get('age'))
        hypertension = int(request.form.get('hypertension'))
        disease = int(request.form.get('disease'))
        work = int(request.form.get('work'))
        glucose = float(request.form.get('glucose'))
        bmi = float(request.form.get('bmi'))
        smoking = int(request.form.get('smoking'))

        input_query = np.array([[sex, age, hypertension, disease, work, glucose, bmi, smoking]])

        print("Received input:", input_query)  # Add this line for debugging

        result = model.predict(input_query)[0]
        return jsonify({'placement': str(result)})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context='adhoc')

