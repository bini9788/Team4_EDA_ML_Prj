import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import OneHotEncoder
import flask
flask.__version__

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/EDA')
def EDA():
    return render_template('EDA.html')

@app.route('/modeling')
def modeling():
    return render_template('modeling.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
