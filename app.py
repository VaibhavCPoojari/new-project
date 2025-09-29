import flask
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application=Flask(__name__)

app=application


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        data_df=data.get_data_as_data_frame()
        print(data_df)
        model_predictor=PredictionPipeline()
        pred=model_predictor.predict(data_df)
        print(pred)
        return render_template('home.html',results=pred[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)  
        