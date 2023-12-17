import os
import pickle as pk
from flask import Flask, request, jsonify
import numpy as np

# Use os.path.join to create a path that works on both Windows and Heroku
model_path = os.path.join('C:\\Users\\Anik\\PycharmProjects\\ML\\models', 'model')

# Check if the file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = pk.load(open(model_path, 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    sex = int(request.form.get('sex'))
    age = int(request.form.get('age'))
    hypertension = int(request.form.get('hypertension'))
    disease = int(request.form.get('disease'))
    work = int(request.form.get('work'))
    glucose = float(request.form.get('glucose'))
    bmi = float(request.form.get('bmi'))
    smoking = int(request.form.get('smoking'))

    input_query = np.array([[sex, age, hypertension, disease, work, glucose, bmi, smoking]])
    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
