import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import gridsearchcv regressor and standard scaler pickle
grid_search_cv=pickle.load(open('Models/gridsearchcv.pkl','rb'))
standard_scaler=pickle.load(open('Models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Age=float(request.form.get('Age'))
        sex = float(request.form.get('sex'))
        cp = float(request.form.get('cp'))
        trestbps = float(request.form.get('trestbps'))
        chol = float(request.form.get('chol'))
        fbs = float(request.form.get('fbs'))
        restecg = float(request.form.get('restecg'))
        thalach = float(request.form.get('thalach'))
        exang = float(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = float(request.form.get('slope'))
        ca = float(request.form.get('ca'))
        thal = float(request.form.get('thal'))
        
        

        new_data_scaled=standard_scaler.transform([[Age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        result=grid_search_cv.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

        
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")