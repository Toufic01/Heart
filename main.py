import pickle as pk
from flask import Flask,request,jsonify
import numpy as np

model = pk.load(open(r'C:\Users\Anik\Documents\model','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods =['POST'])
def predict():
    sex = request.form.get('sex')
    age = request.form.get('age')
    hypertension = request.form.get('hypertension')
    disease = request.form.get('disease')
    work = request.form.get('work')
    glucose = request.form.get('glucose')
    bmi = request.form.get('bmi')
    smoking = request.form.get('smoking')
    input_query = np.array([[sex,age,hypertension,disease,work,glucose,bmi,smoking]])
    result = model.predict(input_query)[0]
    return jsonify({'placement':str(result)})
if __name__ == '__main__':
    app.run(debug=True)