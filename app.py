# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 19:32:50 2021

@author: manisha
"""

import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    if output==0:
        txt='Congrats You do not have diabetes'
    else:
        txt='Opps! You have diabetes'
    
    return render_template('index.html', prediction_text=txt)

if __name__=='__main__':
    app.run(debug=True)
