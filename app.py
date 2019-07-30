#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, url_for, request
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.externals import joblib
from nltk.corpus import stopwords


#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
def home():
	return render_template('index.html')


#prediction function
def reviewPredictor(to_predict_list):
    to_predict = np.array(to_predict_list)
    print(to_predict)
    svmmodel = open('svm_model.pkl','rb')
    clf = joblib.load(svmmodel)
    vectorizer =joblib.load('vectorizer.pkl')
    print (clf)
    reviews_new = vectorizer.transform(to_predict.ravel())         # turn text into count vector
    #reviews_new_tfidf = reviewTF.transform(reviews_new)  # turn into tfidf vector
    print(reviews_new)
    result = clf.predict(reviews_new)
    print(result)
    print(result[0])
    return result[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_review = request.form['review']
        print(to_predict_review)
        print(to_predict_review)
        new_review_arr=[to_predict_review]
        print(new_review_arr)
        result = reviewPredictor(new_review_arr)
        if (result)==2:
           prediction='Prediction is positive'
           print("Prediction is positive")
        elif (result)==1:
           prediction='Prediction is neutral'
           print("Prediction is neutral")
        else:
           prediction='Prediction is negative' 
           print("Prediction is negative")
            
    return render_template("result.html",prediction=prediction)



if __name__ == "__main__":
    app.run(debug=True)
