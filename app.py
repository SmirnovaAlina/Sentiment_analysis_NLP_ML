from flask import Flask,render_template,url_for,request,jsonify
from sklearn.model_selection import train_test_split
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import numpy as np
import joblib
import nltk
# from flask_cors import CORS

app = Flask(__name__)

def reviewPredictor(to_predict_list):
    to_predict = np.array(to_predict_list)
    svmmodel = open('svm_model.pkl','rb')
    clf = joblib.load(svmmodel)
    vectorizer =joblib.load('vectorizer.pkl')
    reviews_new = vectorizer.transform(to_predict.ravel())         
    result = clf.predict(reviews_new)
    return result[0]


@app.route('/')
def home():
    return render_template('index.html')
     


@app.route('/predict',methods=['POST'])
def predict():
 
    
    NB_model = open('NB_model.pkl','rb')
    clf = joblib.load(NB_model)

    vzer = open('vectorizer_NB.pkl','rb')
    review_vzer = joblib.load(vzer)
    
    if request.method == 'POST':
       
        message = request.form['message']
        data = [message]
        # Naive_Bayes model
        tf_idf_vect = review_vzer.transform(data)  
        list_for_dict =[]          
        prediction_dict_NB = dict()
        my_prediction = clf.predict(tf_idf_vect)
        # my_prediction = my_prediction.tolist()[0]
        prediction_dict_NB['model'] = "Naive Bayes"
        prediction_dict_NB['prediction'] = my_prediction.tolist()[0]
        list_for_dict.append(prediction_dict_NB)

        # SVM_model 
        prediction_dict_SVM = dict()
        result = reviewPredictor(data)
        if (result)==2:
        #    prediction='Prediction is positive'
            prediction_dict_SVM['model'] = "Support Vector Machine"
            prediction_dict_SVM['prediction'] = "positive"
        elif (result)==1:
        #    prediction='Prediction is neutral'
            prediction_dict_SVM['model'] = "Support Vector Machine"
            prediction_dict_SVM['prediction'] = "neutral"        
        else:
        #    prediction='Prediction is negative' 
            prediction_dict_SVM['model'] = "Support Vector Machine"
            prediction_dict_SVM['prediction'] = "negative" 
        
        list_for_dict.append(prediction_dict_SVM)

    return render_template('result.html', prediction = list_for_dict, text = message)
    # return jsonify(list_for_dict)

if __name__ == '__main__':
    app.run(debug=True)