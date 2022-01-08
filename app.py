from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    pregnancies = int(request.form.get('pregnancies'))
    glucose = int(request.form.get('glucose'))
    bmi = float(request.form.get('bmi'))
    diabetes_pedigree_function = float(request.form.get('diabetes_pedigree_function'))
    age = int(request.form.get('age'))

    # prediction
    result = model.predict(np.array([pregnancies,glucose,bmi,diabetes_pedigree_function,age]).reshape(1,5))

    if result[0] == 1:
        result = 'diabetic patient'
    else:
        result = 'healthy patient'

    return result

if __name__ == '__main__':
    app.run(debug=True)