from django.shortcuts import render,HttpResponse,redirect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pycaret
from pycaret.datasets import get_data
from pycaret.classification import *
def view(request):
    return render(request, 'index.html')
def get_user_input(request):
# Load and preprocess the diabetes dataset
    data = pd.read_csv('heart.csv')
# preprocess the dataset as needed
    X = np.array(data.drop(columns=['target'], axis=1))
    Y = np.array(data['target'])
# Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
    # Data standardization
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)

# Pycaret
    exp = setup(data=data, target='target',train_size=0.70, session_id=123)
    best_model = compare_models(fold=4)

# check the final params of best model
    best_model.get_params()
    evaluate_model(best_model)
    tuned_model = tune_model(best_model)
    predictions = predict_model(tuned_model, data=data.drop('target', axis = 1, inplace = True))
#Train a Model
    print(tuned_model)
    tuned_model.fit(X_train,Y_train)
    tuned_model.score(X_train,Y_train)
    tuned_model.score(X_test,Y_test)
    acc = tuned_model.score(X_test,Y_test)
    print(acc)



# Define a function to take user input

    if request.method == "POST":
        age = float(request.POST.get('age', 0))
        sex = float(request.POST.get('sex', 0))
        cp = float(request.POST.get('cp', 0))
        trestbps = float(request.POST.get('trestbps', 0))
        chol =float(request.POST.get('chol', '0'))
        fbs = float(request.POST.get('fbs', '0'))
        restecg = float(request.POST.get('restecg', 0))
        thalach = float(request.POST.get('thalach', 0))
        exang = float(request.POST.get('exang', 0))
        oldpeak = float(request.POST.get('oldpeak', 0))
        slope = float(request.POST.get('slope', 0))
        ca = float(request.POST.get('ca', 0))
        thal = float(request.POST.get('thal', 0)) 
# Call your input function to get user input
    user_input =  np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                              exang, oldpeak, slope, ca, thal]])
    prediction = tuned_model.predict(user_input)
    if prediction == [0]:
        # return HttpResponse("The predicted outcome : CONGRATULATIONS!!!, You don`t have any Heart Disease.")
        return render(request, 'HeartDis.html')
    else:
        # return HttpResponse("The predicted outcome : You may have some heart disease") 
        return render(request, 'NoHeartDis.html')
    
    
    
    

    



# Create your views here.
