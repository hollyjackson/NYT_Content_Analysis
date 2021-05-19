#!/usr/bin/env python3

'''
File name: generate_training_set.py
Author: Holly Jackson
Date last modified: 5/19/2021
Python Version: 3.8

This script provides a user interface to generate a training set
from a sample of 500 random articles on Palestine
and/or Israel from the NYT during the First and Second Intifadas. 

It should be run after preprocessing.py.
'''

import os       # standard libraries
import random
import csv
import re
import string

import numpy as np  # numpy

from concrete.util import CommunicationReader # to process the concretely annotated new york times files

# NLTK imports
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize 

# spaCy imports
import spacy
from spacy.symbols import nsubj, VERB
spacy.prefer_gpu()

nlp = spacy.load("en_core_web_lg")

stop_words = set(stopwords.words('english')) 

lemmatizer = WordNetLemmatizer()

## ------------------------------------------------------------------------------------------------
##
## The following code is adapted from
##
## Pentapalli, Nikhil. “Text Cleaning in Natural Language Processing(NLP).” Medium, June 1, 2020.
## https://medium.com/analytics-vidhya/text-cleaning-in-natural-language-processing-nlp-bea2c27035a6
##
##

def clean_text(text):
    # will replace the html characters with " "
    text = re.sub('<.*?>', ' ', text)  
    # to remove the punctuations
    text = text.translate(str.maketrans(' ',' ',string.punctuation))
    # will consider only alphabets and numerics
    text = re.sub('[^a-zA-Z]',' ',text)  
    # will replace newline with space
    text = re.sub("\n"," ",text)
    # will convert to lower case
    text = text.lower()
    return text

def preprocess(text):
    # split into words
    text = clean_text(text)
    # separate into words and remove stop words
    word_tokens = word_tokenize(text) 
    output_text = [w for w in word_tokens if not w in stop_words] 
    # lemmatize words
    lemmatized_text = [lemmatizer.lemmatize(w) for w in output_text]
    return lemmatized_text

##
##
## ------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    # a conservative word bank of phrases that may identify an Israeli or Palestinian subject
    palestine_word_bank = ['Palestine', 'Palestinian', 'Palestinians',
                           'PLO', 'P.L.O.', 'Fatah']
    israel_word_bank = ['Israel', 'Israeli', 'Israelis',
                        'IDF', 'I.D.F.']

    periods = [('first_intifada', range(1987, 1994)), ('second_intifada', range(2000, 2006))]
    articles_sampled = set()
    num_samples = 500

    wordfreq = {}

    already_negative = set()
    already_neutral = set()

    for i in range(num_samples):

        found_valid_path = False

        while not found_valid_path:
        
            # randomly choose an intifada, year, and month
            intifada, years = random.choice(periods)
            year = random.choice(years)
            month = random.choice(range(1,13))

            if month < 10:
                str_month = '0' + str(month)
            else:
                str_month = str(month)
            year_month = str(year) + str_month
            path = "sorted_files/" + intifada + "/" + year_month
            
            # check if this is a valid path
            try:
                _, __, files = next(os.walk(path))
                found_valid_path = True
                break
            except:
                print("Cannot find directory "+year_month+ "...continuing to next date")
                continue
            
        # randomly choose an article from the randomly selected intifada, year, and month
        article = random.choice(files)
        articles_sampled.add(article)

        # parse the text of the article
        comm_file = path + '/' + article
        try:
            for (comm, filename) in CommunicationReader(comm_file):
                comm_text = comm.text
        except:
            continue

        # tokenize the sentences
        sentences = nltk.tokenize.sent_tokenize(comm_text)

        # go through and split sentences
        sentences_copy = []
        for phrase in sentences:
            sentences_copy.extend(list(filter(lambda x: x != '', phrase.split("\n"))))

        sentences = sentences_copy


        # find all subjects mentioned in the article & compile their relevant associations
        article_dictionary = {}

        for sentence in sentences:

            # parse sentence using spacy nlp library
            nlp_sentence = nlp(sentence)

            for token in nlp_sentence:
                if token.dep_ == "nsubjpass" or token.dep_ == "csubjpass" or token.dep_ == "nsubj" or token.dep_ == "csubj":

                    related_descriptions = []
                    for j in range(token.i+1, token.head.i):
                        related_descriptions.append(nlp_sentence[j].text)

                    # find all words in between token and token.head
                    
                    if token.text not in article_dictionary.keys():
                        article_dictionary[token.text] = set()

                    # add any clauses in between
                    for word in related_descriptions:
                        article_dictionary[token.text].add(word)

                    # add relevant adjectives
                    for child in token.children:
                        article_dictionary[token.text].add(child.text)

        # iterate through each sentence & detect tone/voice
        for sentence in sentences:

            # parse sentence using spacy nlp library
            nlp_sentence = nlp(sentence)

            for token in nlp_sentence:

                if token.dep_ == "nsubjpass" or token.dep_ == "csubjpass" or token.dep_ == "nsubj" or token.dep_ == "csubj":

                    # check if the subject is Israeli, Palestinian, or neither
                    palestine_relevance = token.text in palestine_word_bank
                    israel_relevance = token.text in israel_word_bank
                    if not israel_relevance and not palestine_relevance:
                        palestine_relevance = any([word in article_dictionary[token.text] for word in palestine_word_bank])
                        israel_relevance = any([word in article_dictionary[token.text] for word in israel_word_bank])
                    
                    # if relevant to either, note this word as a relevant word for the training set
                    if palestine_relevance or israel_relevance:
                        if token.head.pos_ == "VERB" or token.head.pos_ == "ADJ":
                            if token.head.text in wordfreq.keys():
                                wordfreq[token.head.text] += 1
                            else:
                                wordfreq[token.head.text] = 1

    # sort the words by their frequency
    wordfreq = dict(sorted(wordfreq.items(), key=lambda item: item[1]))

    # prompt the user to identify words
    print('Get ready to identify', len(wordfreq), 'words')

    # write results to CSV file
    with open('training_data.csv', mode='w') as training_file:
        writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['text', 'class_label', 'freq'])

        num_words = len(wordfreq.keys())
        keys = list(wordfreq.keys())

        violent_words = set()

        for word in wordfreq.keys():
            print(word)
        
            # user will be prompted to input a 1 if the word conveys violence
            # and any other character otherwise
            violent = input("Does this verb covey violence?  ")
            
            if violent == '1':
                violent_words.add(word)

        for word in wordfreq.keys():
            if word in violent_words:
                writer.writerow([word, '1', str(wordfreq[word])])
            else:
                writer.writerow([word, '0', str(wordfreq[word])])
