from flask import Flask,render_template,url_for,request,jsonify
from sklearn.model_selection import train_test_split
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk
# from flask_cors import CORS

app = Flask(__name__)


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
        tf_idf_vect = review_vzer.transform(data)            
        prediction_dict = dict()
        my_prediction = clf.predict(tf_idf_vect)
        # my_prediction = my_prediction.tolist()[0]
        prediction_dict['Naive_Bayes'] = my_prediction.tolist()[0]
       
    return render_template('result.html', prediction = prediction_dict)
    # return jsonify(prediction_dict)

if __name__ == '__main__':
    app.run(debug=True)