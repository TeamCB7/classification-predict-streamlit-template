"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import spacy

# Vectorizer
news_vectorizer = open("resources/vectorizer (2).pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
raw.head()

# Load model from path
def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information","Prediction", "NLP" ]
	selection = st.sidebar.selectbox("Choose Option", options)


	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("In this app we will be classifying twitter sentiments around climate change. For the sake of transparency we have shared the data upon which our algorithms have trained ")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		all_ml_models = ["LogisticRegression","XGB","Linear SVC"]
		model_choice = st.selectbox("Select Model",all_ml_models)

		prediction_labels = {'negative': -1,'neutral': 0,'positive': 1,'news': 2}
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
		if st.button("Classify"):
			st.text("Original Text::\n{}".format(tweet_text))
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			if model_choice == 'LogisticRegression':
				predictor = load_prediction_models("resources/log_reg (2).pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)

			elif model_choice == 'XGB':
				predictor = load_prediction_models("resources/xgb.pkl")
				prediction = predictor.predict(vect_text)
			
			elif model_choice == 'Linear SVC':
				predictor = load_prediction_models("resources/svc.pkl")
				prediction = predictor.predict(vect_text)

			final_result = get_key(prediction,prediction_labels)
			st.success("Text Categorized as:: {}".format(final_result))

			# When model has successfully run, will print prediction
			st.success("Text Categorized as: {}".format(prediction))
	
	if selection == 'NLP':
		st.info("Natural Language Processing of Text")
		raw_text = st.text_area("Enter Text Here","Type Here")
		nlp_task = ["Tokenization","Lemmatization","NER","POS Tags"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("Original Text::\n{}".format(raw_text))

			docx = nlp(raw_text)
			if task_choice == 'Tokenization':
				result = [token.text for token in docx ]
			elif task_choice == 'Lemmatization':
				result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
			elif task_choice == 'NER':
				result = [(entity.text,entity.label_)for entity in docx.ents]
			elif task_choice == 'POS Tags':
				result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]

			st.json(result)

		if st.button("Tabulize"):
			docx = nlp(raw_text)
			c_tokens = [token.text for token in docx ]
			c_lemma = [token.lemma_ for token in docx ]
			c_pos = [token.pos_ for token in docx ]

			new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
			st.dataframe(new_df)


		if st.checkbox("WordCloud"):
			c_text = raw_text
			wordcloud = WordCloud().generate(c_text)
			plt.imshow(wordcloud,interpolation='bilinear')
			plt.axis("off")
			st.pyplot()



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
