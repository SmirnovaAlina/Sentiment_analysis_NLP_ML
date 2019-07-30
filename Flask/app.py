from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

# function to prepare text to model 

# def text_prep(raw_text):
#     review_vzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
#     data_train_counts = review_vzer.fit_transform(raw_text)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
	# Use our Saved Model

	NB_model = open('NB_model.pkl','rb')
	clf = joblib.load(NB_model)
    
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = review_vzer.transform(data) 
        tf_idf_vect = reviewTfmer.transform(vect)
		my_prediction = clf.predict(tf_idf_vect)

	return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)