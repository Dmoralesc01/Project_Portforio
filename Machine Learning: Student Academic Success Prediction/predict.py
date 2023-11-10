import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'FinalModel.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
app = Flask('student_dropout_predictor')

@app.route('/predict', methods = ['POST'])
def predict():
    student = request.get_json()
    
    X = dv.transform([student])
    y_pred = model.predict(X)[0]
    
    result = {'Result':y_pred}
    
    return jsonify(result)

if __name__ =='__main__':
    app.run(debug=True, host = '0.0.0.0', port = 9696)