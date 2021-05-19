#!/usr/bin/env python3

'''
File name: violence_count.py
Author: Holly Jackson
Date last modified: 5/19/2021
Python Version: 3.8

This script articles counts references to violence in articles
processed by my NLP pipeline.

It should be run after preprocessing.py and voice_identifier.py.
'''

import os       # standard libraries
import sys
import re
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# some useful nlp imports 
import keras
import nltk
import pandas as pd
import codecs

# scikit-learn imports
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from concrete.util import CommunicationReader # to process the concretely annotated new york times files

## ------------------------------------------------------------------------------------------------
##
## The following code is adapted from
##
## Jagota, Arun. “Named Entity Recognition in NLP.” Medium. Towards Data Science, October 14, 2020.
## https://towardsdatascience.com/named-entity-recognition-in-nlp-be09139fa7b8. 
##
##

def generate_model():

    # count vectorizer
    def cv(data):
        count_vectorizer = CountVectorizer()
        emb = count_vectorizer.fit_transform(data)
        return emb, count_vectorizer

    # load csv file with hand-labeled training data
    df = pd.read_csv(r'training_data.csv')
    list_corpus = df['text'].to_list()
    list_labels = df["class_label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, 
                                                                                    random_state=40)
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)

    # tf-idf vectorizer
    def tfidf(data):
        tfidf_vectorizer = TfidfVectorizer()
        train = tfidf_vectorizer.fit_transform(data)
        return train, tfidf_vectorizer

    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # define regression model
    clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                            multi_class='multinomial', n_jobs=-1, random_state=40)
    clf_tfidf.fit(X_train_tfidf, y_train)

    y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)

    def get_metrics(y_test, y_predicted):  
        # true positives / (true positives+false positives)
        precision = precision_score(y_test, y_predicted, pos_label=None,
                                        average='weighted')             
        # true positives / (true positives + false negatives)
        recall = recall_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
        
        # harmonic mean of precision and recall
        f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
        
        # true positives + true negatives/ total
        accuracy = accuracy_score(y_test, y_predicted)
        return accuracy, precision, recall, f1

    # print accuracy metrics of model
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf, 
                                                                       recall_tfidf, f1_tfidf))

    return clf_tfidf, tfidf_vectorizer

##
##
## ------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # generate violence classifier model from hand-trained data 
    clf_tfidf, tfidf_vectorizer = generate_model()
    
    periods = [('first_intifada', range(1987, 1994)), ('second_intifada', range(2000, 2006))]
    
    for intifada, years in periods:
    
        palestine_violent_count = {}
        israel_violence_count = {}

        for year in years:
            for month in range(1, 13):

                # check if valid path
                if month < 10:
                    str_month = '0' + str(month)
                else:
                    str_month = str(month)
                year_month = str(year) + str_month

                path = sys.path[0] +'/content_analysis/' + intifada +'/' + year_month

                try:
                    palestine_PASSIVE = np.load(path + 'palestine_passive.npy', allow_pickle = True)
                    israel_PASSIVE = np.load(path + 'israel_passive.npy', allow_pickle = True)
                    palestine_ACTIVE = np.load(path + 'palestine_active.npy', allow_pickle = True)
                    israel_ACTIVE = np.load(path + 'israel_active.npy', allow_pickle = True)
                except:
                    print("Cannot find files for " + year_month + "...continuing to next date")
                    continue

                # classify violence in passive voice references to Palestinians
                for article in palestine_PASSIVE:
                
                    if article["obj"] != 1:

                        is_violent = int(clf_tfidf.predict(tfidf_vectorizer.transform([article["verb"]]))[0])
                
                        if article["article"] in palestine_violent_count:
                            palestine_violent_count[ article["article"]] += is_violent
                        else:
                            palestine_violent_count[ article["article"]] = is_violent
                        

                # classify violence in active voice references to Palestinians
                for article in palestine_ACTIVE:
                
                    if article["obj"] != 1:

                        is_violent = int(clf_tfidf.predict(tfidf_vectorizer.transform([article["verb"]]))[0])

                        if article["article"] in palestine_violent_count:
                            palestine_violent_count[ article["article"]] += is_violent
                        else:
                            palestine_violent_count[ article["article"]] = is_violent
              

                # classify violence in passive voice references to Israel
                for article in israel_PASSIVE:
                
                    if article["obj"] != 1:

                        is_violent = int(clf_tfidf.predict(tfidf_vectorizer.transform([article["verb"]]))[0])

                        if article["article"] in israel_violence_count:
                            israel_violence_count[ article["article"]] += is_violent
                        else:
                            israel_violence_count[ article["article"]] = is_violent
                

                # classify violence in active voice references to Israel
                for article in israel_ACTIVE:
                
                    if article["obj"] != 1:

                        is_violent = int(clf_tfidf.predict(tfidf_vectorizer.transform([article["verb"]]))[0])

                        if article["article"] in israel_violence_count:
                            israel_violence_count[ article["article"]] += is_violent
                        else:
                            israel_violence_count[ article["article"]] = is_violent

        palestine_violent_count = dict(sorted(palestine_violent_count.items(), key=lambda item: item[1]))
        israel_violence_count = dict(sorted(israel_violence_count.items(), key=lambda item: item[1]))

        # save json files for plotting later
        json.dump(
            palestine_violent_count,
            open("palestine_violent_count_" + intifada + ".json", "w"))
        
        json.dump(
            israel_violence_count,
            open("israel_violence_count_" + intifada + ".json", "w"))

        print(palestine_violent_count)

        print(israel_violence_count)