#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:33:29 2022

@author: sanjasrdanovic
"""

# import packages and libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# NLP
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words("english")
import spacy
nlp = spacy.load('en_core_web_sm')

# Scikit-Learn: 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

# Keras and Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 

# Set tensorflow and numpy random seed for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

#%%

# load dataset from csv
df = pd.read_csv("/Users/sanjasrdanovic/Desktop/Data Science/Machine Learning/Projekt ML/tripadvisor_hotel_reviews.csv")

# data info
print(df.head())
print(df.info()) # 20491 entries 2 columns

print(f"Theres {df.shape[0]} reviews in this dataset.")

#%%

# Visualisation - count of ratings
palette = ["#f31c1c", "#ed9517", "#ffe23b", "#c7ff46", "#17b882"]
sns.set_palette(palette)
sns.countplot(x='Rating', data = df)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Countplot Review Ratings')
plt.show()
plt.savefig("Countplot Review Ratings.png") # save figure   

#%%
# Data Inspection, checking if there are null or NaN values

print(df.isnull().sum())
print(df.isna().sum())
print(df.describe())

# No NaN values

#%%

# Encoding

# define label encode
def label_encode(x):
    if x == 1 or x == 2: 
        return 0
    if x == 3:
        return 1
    if x == 5 or x == 4: 
        return 2
    
# define label to name
def label2name(x):
    if x == 0:
        return "Negative"
    if x == 1:
        return "Neutral"
    if x == 2:
        return "Positive"
    
# encode label and mapping label name, add columns to the data frame
df["label"] = df["Rating"].apply(lambda x: label_encode(x))
df["label_name"] = df["label"].apply(lambda x: label2name(x))

print(df['label'].value_counts())
# 2    15093
# 0     3214
# 1     2184
#%%

# Visualisation - count of ratings
palette = ["#f31c1c", "#ffe23b", "#17b882"]
sns.set_palette(palette)
sns.countplot(x='label', data = df)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Countplot Sentiment')
plt.show()

#%%

# Downsampling

# Shuffle the Dataset.
shuffled_df = df.sample(frac=1,random_state=4)
sample_size = 6552
class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]
class_2 = df[df['label'] == 2]

sample_class = int(sample_size /3)
df = pd.concat([class_0.sample(sample_class), 
                class_1.sample(sample_class), 
                class_2.sample(sample_class)] , axis =0)
print('class_0:', class_0.shape)
print('class_1:', class_1.shape)
print('class_2:', class_1.shape)
      
#plot the dataset after the undersampling
plt.figure(figsize=(8, 8))
sns.countplot('label', data=df)
plt.title('Balanced Classes')
plt.show()

#%%
# PREPROCESSING

# Cleaning Special Characters and Removing Punctuations:
def clean_text(review):
    pattern = r'[^a-zA-z0-9\s]'
    review = re.sub(pattern, '', review)
    return review

# Cleaning digits
def clean_numbers(review):
    review = ''.join([i for i in review if not i.isdigit()])
    return review

# Removing Contractions
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "n't": "not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(review):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, review)

df["Review"] = df["Review"].apply(clean_text).apply(clean_numbers).apply(replace_contractions)

# Remove stopwords
df["Review"] = [word for word in df["Review"] if word not in stopwords]

# Lemmatisierung
df["Review"] = df["Review"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))

#%%

# Tokenization

# df['tokenized_review'] = df['Review'].apply(lambda x: nlp.tokenizer(x))

review = df["Review"].copy() # Use a copy of the clean reviews
token = Tokenizer() # Initialize the tokenizer
token.fit_on_texts(review) # Fit the tokenizer to the reviews
texts = token.texts_to_sequences(review) # Convert the reviews into sequences for keras to use

#%%

# Padding

# Print an example sequence to make sure everything is working
print("Into a Sequence: ")
print(texts[26])

texts = pad_sequences(texts, padding='post') # Pad the sequences to make them similar lengths

# Print an example padded sequence to make sure everything is working
print("After Padding: ")
print(texts[26])

#%%

X = texts # Input values.
y = df['label']   # Output values 

# y=to_categorical(y,)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test: ' + str(X_test.shape))
print('Y_test: ' + str(y_test.shape))

print(type(X_test))
#%%

# Learning Classifiers

dt_clf = DecisionTreeClassifier(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42) #n_estimators=100, max_depth=10,
svm_clf = SVC(gamma="scale", random_state=42, probability=True)

# hard voting

voting = VotingClassifier(
             estimators=[('dt', dt_clf),
                         ('rf', rnd_clf),
                         ('svc', svm_clf)], 
             voting='hard',
             )
parameters = {

    "rf__n_estimators":[5,10,50,100,250],
    "rf__max_depth":[2,4,8,16,32,None],

}

voting.fit(X_train,y_train)

for clf, label in zip([dt_clf, rnd_clf, svm_clf, voting], ['Decision Tree', 'Random Forest', 'SVC', 'Ensemble']):
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5, error_score="raise")
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    # will compute the individual accuracy of each model

#%%

# soft voting

# Learning Classifiers

dt_clf = DecisionTreeClassifier(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42) #n_estimators=100, max_depth=10,
svm_clf = SVC(gamma="scale", random_state=42, probability=True)

voting = VotingClassifier(
             estimators=[('dt', dt_clf),
                         ('rf', rnd_clf),
                         ('svc', svm_clf)], 
             voting='soft',
             flatten_transform=False)

dt_clf = dt_clf.fit(X_train, y_train)
rnd_clf = rnd_clf.fit(X_train, y_train)
svm_clf = svm_clf .fit(X_train, y_train)
voting= voting.fit(X_train, y_train)

parameters = {
    "rf__n_estimators":[5,10,50,100,250],
    "rf__max_depth":[2,4,8,16,32,None],
}

for clf in (dt_clf, rnd_clf, svm_clf, voting):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

voting.get_params()

#%%
# Classification reports and confusion matrix 

# Descision Tree

# compare y_test and y_pred
y_pred = dt_clf.predict(X_test)
y_pred_dt = np.around(y_pred)

# Confusion Matrix

cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)

# Classification Report

classrep_dt = metrics.classification_report(y_test,y_pred_dt)
print(classrep_dt)

heatmap_dt = pd.DataFrame(data=cm_dt, columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'], 
                        index=['Predict Negative', 'Predict Neutral', 'Predict Positive'])
sns.heatmap(heatmap_dt, annot=True, fmt='d', cmap='YlGnBu')
plt.show()
#%%

# Random Forest

# compare y_test and y_pred
y_pred = rnd_clf.predict(X_test)
y_pred_rnd = np.around(y_pred)
# print(y_pred.shape)
# print(y_pred_rnd.shape)
# print(y_test.shape)
# np.unique(y_pred)
# np.unique(y_pred_rnd)

# Confusion Matrix

cm_rnd = confusion_matrix(y_test, y_pred_rnd)
print(cm_rnd)

# Classification Report
# metrics.f1_score(y_test, y_pred_rnd, average='weighted', labels=np.unique(y_pred_rnd))

classrep_rnd = metrics.classification_report(y_test,y_pred_rnd)
print(classrep_rnd)

heatmap_rnd = pd.DataFrame(data=cm_rnd, columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'], 
                        index=['Predict Negative', 'Predict Neutral', 'Predict Positive'])
sns.heatmap(heatmap_rnd, annot=True, fmt='d', cmap='YlGnBu')
plt.show()



#%%
# SVC

# compare y_test and y_pred
y_pred = svm_clf.predict(X_test)
y_pred_svm = np.around(y_pred)

# Confusion Matrix

cm_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_svm)

# Classification Report

classrep_svm = metrics.classification_report(y_test,y_pred_svm)
print(classrep_svm)

heatmap_svm = pd.DataFrame(data=cm_svm, columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'], 
                        index=['Predict Negative', 'Predict Neutral', 'Predict Positive'])
sns.heatmap(heatmap_svm, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

#%%

# Voting

# compare y_test and y_pred
y_pred=voting.predict(X_test)
y_pred = np.around(y_pred)

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report

modelrep = metrics.classification_report(y_test,y_pred)
print(modelrep)

heatmap_voting = pd.DataFrame(data=cm, columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'], 
                        index=['Predict Negative', 'Predict Neutral', 'Predict Positive'])
sns.heatmap(heatmap_voting, annot=True, fmt='d', cmap='YlGnBu')
plt.show()
#%%
# Let's predict on few reviews
neg_review = ['Rooms were old and not very clean. Nothing like in pictures. Staff difficult to reach. \
              Food bad. Loud room parties. Pick another hotel. This is definitely on a low level']

# Let's tokenize it and do the pad_sequence to make it in right format acceptable by model
neg_review_token = token.texts_to_sequences(neg_review)

# padding
neg_review_padded = pad_sequences(neg_review_token,maxlen=1911,padding='post')
review_predict = (voting.predict(neg_review_padded)>0.5).astype('int32')

# 0 is negative review, 1 is neutral and else is positive
if review_predict[0] == 0:
    print("It's a negative review")
elif review_predict[0] == 1:
    print("It's a neutral review")
else:
    print("It's a positive review")
    
# Let's try another one.This time we will take a positive review
pos_review = ["It was beautiful, and the staff was very friendly. The rooms are clean and modern. \
              Very impressed with this hotel!"]

# Tokenization
pos_review = token.texts_to_sequences(pos_review)

# padding
pos_review = pad_sequences(pos_review,maxlen=1911,padding='post')

# prediction
review_predict = (voting.predict(pos_review)>0.5).astype('int')

# 0 is negative review, 1 is neutral and else is positive
if review_predict[0] == 0:
    print("It's a negative review")
elif review_predict[0] == 1:
    print("It's a neutral review")
else:
    print("It's a positive review")

#%%

# VADER Sentiment Analysis 
# based on the sentiment on the textual data -reviews, now on ratings the customers gave (1-5), i.e., labels (0,1,2)

# print(df['Rating'].value_counts())
# 5    9054
# 4    6039
# 3    2184
# 2    1793
# 1    1421
# df.dropna(inplace=True)
# print(df['label'].value_counts())
# 2    15093
# 0     3214
# 1     2184

# df.loc[234]['Review'] # first review

sid = SentimentIntensityAnalyzer()
# print(sid.polarity_scores(df.loc[234]['Review']))

# {'neg': 0.072, 'neu': 0.643, 'pos': 0.285, 'compound': 0.9747} -> Positive

df['scores'] = df['Review'].apply(lambda review: sid.polarity_scores(review))
df.head()

df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])
df.head()

df['comp_score'] = df['compound'].apply(lambda c: 'Positive' if c >=0.05 
                                        else ( 'Negative' if c<= - 0.05 
                                              else 'Neutral'))

print(df.head())

print(accuracy_score(df['label_name'],df['comp_score']))

print(classification_report(df['label_name'],df['comp_score']))

print(confusion_matrix(df['label_name'],df['comp_score']))

heatmap_vader = pd.DataFrame(data=confusion_matrix(df['label_name'],df['comp_score']), columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'], 
                        index=['Predict Negative', 'Predict Neutral', 'Predict Positive'])
sns.heatmap(heatmap_vader, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

#%%






