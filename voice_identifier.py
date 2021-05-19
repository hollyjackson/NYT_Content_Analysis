#!/usr/bin/env python3

'''
File name: voice_identifier.py
Author: Holly Jackson
Date last modified: 5/19/2021
Python Version: 3.8

This script processes articles related to Israel and/or Palestine
in the Concretely Annotated New York Times.

It should be run after preprocessing.py (with all data in sorted_files).
'''

import os       # standard libraries
import argparse

from tqdm import tqdm   # 3rd party libraries
import numpy as np

from numba import jit, njit, prange # for cpu parallelization

# NLTK imports
import nltk
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('wordnet')
from nltk.corpus import sentiwordnet as swn
from nltk.stem.wordnet import WordNetLemmatizer

# spaCy imports
import spacy
from spacy.lang.en import English
from spacy.symbols import nsubj, VERB
from spacytextblob.spacytextblob import SpacyTextBlob
spacy.prefer_gpu()

from concrete.util import CommunicationReader # to process the concretely annotated new york times files

# load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('spacytextblob')

# enumerate spacy subject types
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]


class NLPAnalyzer:

    def __init__(self):
        # a conservative word bank of phrases that may identify an Israeli or Palestinian subject
        self.palestine_word_bank = ['Palestine', 'Palestinian', 'Palestinians', 'PLO', 'P.L.O.', 'Fatah']
        self.israel_word_bank = ['Israel', 'Israeli', 'Israelis', 'IDF', 'I.D.F.']

    def detect_passive_voice(self, sentence):
        # detect passive voice using parse tree and token identifiers
        sent = list(nlp(sentence).sents)[0]
        if sent.root.tag_ == "VBN" or sent.root.tag_ == "VBG":
            return (True, sent.root)
        for w in sent.root.children:
            if w.dep_ == "aux" and (w.tag_ == "VBN" or w.tag_ == "VBG"):
                return (True, w)
        return(False, None)

    def analyze_tone_voice(self, path):

        palestine_passive = []
        israel_passive = []

        palestine_active = []
        israel_active = []

        _, __, files = next(os.walk(path))
        file_count = len(files)

        for i in tqdm(range(file_count)):

            article = files[i]
            
            found = False
            comm_file = path + '/' + article

            # try to read & open concrete file
            try:
                for (comm, filename) in CommunicationReader(comm_file):
                    comm_text = comm.text
            except:
                continue

            # tokenize the text
            sentences = nltk.tokenize.sent_tokenize(comm_text)

            # go through and split
            sentences_copy = []
            for phrase in sentences:
                sentences_copy.extend(list(filter(lambda x: x != '', phrase.split("\n"))))
            sentences = sentences_copy

            # record voice examples for passive & active voice 
            article_palestine_passive = []
            article_israel_passive = []

            article_palestine_active = []
            article_israel_active = []

            # find all subjects mentioned in the article & compile their relevant associations
            article_dictionary = {}
            
            for sentence in sentences:

                # parse sentence
                nlp_sentence = nlp(sentence)

                for token in nlp_sentence:
                    if token.dep_ in SUBJECTS:

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

                # detect passive voice in sentence
                passive, verb = self.detect_passive_voice(sentence)

                # use nlp package from spacy to parse sentence
                nlp_sentence = nlp(sentence)

                # loop through tokens in parsed sentence
                for token in nlp_sentence:

                    # -------------------------------
                    # HANDLE PASSIVE VOICE REFERENCES
                    # -------------------------------

                    if (token.dep_ == "nsubjpass" or token.dep_ == "csubjpass") and passive and token.head.text == verb.text:

                        # check if the subject is Israeli, Palestinian, or neither
                        palestine_relevance = token.text in self.palestine_word_bank
                        israel_relevance = token.text in self.israel_word_bank
                        if not israel_relevance and not palestine_relevance:
                            palestine_relevance = any([word in article_dictionary[token.text] for word in self.palestine_word_bank])
                            israel_relevance = any([word in article_dictionary[token.text] for word in self.israel_word_bank])

                        # --------------------------------------------
                        # HANDLE PALESTINIAN SUBJECTS -- PASSIVE VOICE
                        # --------------------------------------------

                        if palestine_relevance and not israel_relevance:

                            # classify sentiment of word using SentiWordNet
                            if token.head.pos_ == "VERB":
                                words = swn.senti_synsets(token.head.text, 'v')
                            elif token.head.pos_ == "ADJ":
                                words = swn.senti_synsets(token.head.text, 'a')
                            else:
                                continue

                            list_words = list(words)
                            if len(list_words) > 0:
                                # save sentence data (subject/word/voice/sentiment)
                                word1 = list_words[0]
                                sentence_data = {"article": article,
                                                "year" : year,
                                                "month" : month,
                                                "pos" : word1.pos_score(),
                                                "neg" : word1.neg_score(),
                                                "obj" : word1.obj_score(),
                                                "voice" : "passive",
                                                "relevance" : "palestine",
                                                "subj" : token.text,
                                                "verb" : token.head.text}
                                article_palestine_passive.append(sentence_data)
                                palestine_passive.append(sentence_data)

                        # ----------------------------------------
                        # HANDLE ISRAELI SUBJECTS -- PASSIVE VOICE
                        # ----------------------------------------

                        elif israel_relevance and not palestine_relevance:
                            
                            # classify sentiment of word using SentiWordNet
                            if token.head.pos_ == "VERB":
                                words = swn.senti_synsets(token.head.text, 'v')
                            elif token.head.pos_ == "ADJ":
                                words = swn.senti_synsets(token.head.text, 'a')
                            else:
                                continue
                            
                            list_words = list(words)
                            if len(list_words) > 0:  
                                # save sentence data (subject/word/voice/sentiment)   
                                word1 = list_words[0]
                                sentence_data = {"article": article,
                                                "year" : year,
                                                "month" : month,
                                                "pos" : word1.pos_score(),
                                                "neg" : word1.neg_score(),
                                                "obj" : word1.obj_score(),
                                                "voice" : "passive",
                                                "relevance" : "israel",
                                                "subj" : token.text,
                                                "verb" : token.head.text}
                                article_israel_passive.append(sentence_data)
                                israel_passive.append(sentence_data)
                        else:
                            # discard irrelavent or unresolved subjects
                            continue



                    # ------------------------------
                    # HANDLE ACTIVE VOICE REFERENCES
                    # ------------------------------

                    if (token.dep_ == "nsubj" or token.dep_ == "csubj") and not passive:

                        # check if the subject is Israeli, Palestinian, or neither
                        palestine_relevance = token.text in self.palestine_word_bank
                        israel_relevance = token.text in self.israel_word_bank
                        if not israel_relevance and not palestine_relevance:
                            palestine_relevance = any([word in article_dictionary[token.text] for word in self.palestine_word_bank])
                            israel_relevance = any([word in article_dictionary[token.text] for word in self.israel_word_bank])

                        # -------------------------------------------
                        # HANDLE PALESTINIAN SUBJECTS -- ACTIVE VOICE
                        # -------------------------------------------
                        
                        if palestine_relevance and not israel_relevance:

                            # save sentence data (subject/word/voice/sentiment) 
                            if token.head.pos_ == "VERB":
                                words = swn.senti_synsets(token.head.text, 'v')
                            elif token.head.pos_ == "ADJ":
                                words = swn.senti_synsets(token.head.text, 'a')
                            else:
                                continue
                            
                            list_words = list(words)
                            if len(list_words) > 0:   
                                # save sentence data (subject/word/voice/sentiment)   
                                word1 = list_words[0]
                                sentence_data = {"article": article,
                                                "year" : year,
                                                "month" : month,
                                                "pos" : word1.pos_score(),
                                                "neg" : word1.neg_score(),
                                                "obj" : word1.obj_score(),
                                                "voice" : "active",
                                                "relevance" : "palestine",
                                                "subj" : token.text,
                                                "verb" : token.head.text}
                                article_palestine_active.append(sentence_data)
                                palestine_active.append(sentence_data)

                        # ---------------------------------------
                        # HANDLE ISRAELI SUBJECTS -- ACTIVE VOICE
                        # ---------------------------------------

                        elif israel_relevance and not palestine_relevance:

                            # save sentence data (subject/word/voice/sentiment) 
                            if token.head.pos_ == "VERB":
                                words = swn.senti_synsets(token.head.text, 'v')
                            elif token.head.pos_ == "ADJ":
                                words = swn.senti_synsets(token.head.text, 'a')
                            else:
                                continue
                            
                            list_words = list(words)
                            if len(list_words) > 0:    
                                # save sentence data (subject/word/voice/sentiment)  
                                word1 = list_words[0]
                                sentence_data = {"article": article,
                                                "year" : year,
                                                "month" : month,
                                                "pos" : word1.pos_score(),
                                                "neg" : word1.neg_score(),
                                                "obj" : word1.obj_score(),
                                                "voice" : "active",
                                                "relevance" : "israel",
                                                "subj" : token.text,
                                                "verb" : token.head.text}
                                article_israel_active.append(sentence_data)
                                israel_active.append(sentence_data)
                                
                        else:
                            # discard irrelavent or unresolved subjects
                            continue

        return palestine_passive, israel_passive, palestine_active, israel_active


if __name__ == "__main__":

    # parse a particular year & intifada

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--intifada", type=str)
    args = parser.parse_args()

    assert (args.intifada == 'first_intifada' or args.intifada == 'second_intifada')
    assert (args.intifada == 'first_intifada' and args.year in range(1987, 1996)) or (args.intifada == 'second_intifada' and args.year in range(2000, 2006))

    intifada = args.intifada
    year = args.year

    analyzer = NLPAnalyzer()

    for month in range(1, 13):

        # check if valid path
        if month < 10:
            str_month = '0' + str(month)
        else:
            str_month = str(month)
        year_month = str(year) + str_month


        path = "sorted_files/" + intifada + "/" + year_month

        try:
            _, __, files = next(os.walk(path))
        except:
            print("Cannot find directory "+year_month+ "...continuing to next date")
            continue

        palestine_passive, israel_passive, palestine_active, israel_active = analyzer.analyze_tone_voice(path)

        np.save('content_analysis/' + intifada +'/' + year_month + 'palestine_passive.npy', np.array(palestine_passive))
        np.save('content_analysis/' + intifada +'/' + year_month + 'israel_passive.npy', np.array(israel_passive))
        np.save('content_analysis/' + intifada +'/' + year_month + 'palestine_active.npy', np.array(palestine_active))
        np.save('content_analysis/' + intifada +'/' + year_month + 'israel_active.npy', np.array(israel_active))