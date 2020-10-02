import flask
from flask import Flask, jsonify, request
import json
from catboost import CatBoostRegressor
import numpy as np

app = Flask(__name__)

def load_models():
    file_name = "models/cat_model"
    model = CatBoostRegressor().load_model(file_name,format='cbm')
    return model

@app.route('/predict', methods=['GET'])
def predict():
    # getting input features
    request_json = request.get_json()
    x = request_json['input']
    # load model
    x = np.array(x)
    model = load_models()
    prediction = model.predict(x)
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == '__main__':
 	application.run(debug=True)