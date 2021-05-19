#!/usr/bin/env python3

'''
File name: voice_count.py
Author: Holly Jackson
Date last modified: 5/19/2021
Python Version: 3.8

This script counts instances of passive and active voice
in articles processed by my NLP pipeline.

It should be run after preprocessing.py and voice_identifier.py.
'''

import os       # standard libraries
import sys
import json

import numpy as np  # numpy


if __name__ == "__main__":
    
    periods = [('first_intifada', range(1987, 1994)),('second_intifada', range(2000, 2006))]#
    
    for intifada, years in periods:

        p_passive_count = {}
        p_active_count = {}

        i_active_count = {}
        i_passive_count = {}

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

                # PASSIVE VOICE -- PALESTINIAN SUBJECT
                for article in palestine_PASSIVE:
                    if article["article"] in p_passive_count:
                        p_passive_count[article["article"]] += 1
                    else:
                        p_passive_count[article["article"]] = 1

                # ACTIVE VOICE -- PALESTINIAN SUBJECT
                for article in palestine_ACTIVE:
                    if article["article"] in p_active_count:
                        p_active_count[article["article"]] += 1
                    else:
                        p_active_count[article["article"]] = 1
                
                # PASSIVE VOICE -- ISRAELI SUBJECT
                for article in israel_PASSIVE:
                    if article["article"] in i_passive_count:
                        i_passive_count[article["article"]] += 1
                    else:
                        i_passive_count[article["article"]] = 1

                # ACTIVE VOICE -- ISRAELI SUBJECT
                for article in israel_ACTIVE:
                    if article["article"] in i_active_count:
                        i_active_count[article["article"]] += 1
                    else:
                        i_active_count[article["article"]] = 1

        # sort by highest occurrence
        i_passive_count = dict(sorted(i_passive_count.items(), key=lambda item: item[1]))
        i_active_count = dict(sorted(i_active_count.items(), key=lambda item: item[1]))
        p_passive_count = dict(sorted(p_passive_count.items(), key=lambda item: item[1]))
        p_active_count = dict(sorted(p_active_count.items(), key=lambda item: item[1]))

        # save json files for plotting later
        json.dump(
            i_passive_count,
            open("i_passive_count" + intifada + ".json", "w"))
        
        json.dump(
            i_active_count,
            open("i_active_count" + intifada + ".json", "w"))

        json.dump(
            p_passive_count,
            open("p_passive_count" + intifada + ".json", "w"))
        
        json.dump(
            p_active_count,
            open("p_active_count" + intifada + ".json", "w"))

        print('INTIFADA ', intifada)

        # find article with highest combined Israeli active voice count and Palestinian passive voice count
        max_combined = 0
        max_article = None
        for article in p_passive_count.keys():
            if article in i_active_count.keys():
                iac = i_active_count[article]
                ipc = i_passive_count[article] if article in i_passive_count.keys() else 0
                ppc = p_passive_count[article]
                pac = p_active_count[article] if article in p_active_count.keys() else 0

                score = iac + ppc
                if score > max_combined:
                    max_combined = score
                    max_article = article

        print(max_article, p_passive_count[max_article], i_active_count[max_article])


        # find article with highest combined Israeli passive voice count and Palestinian active voice count
        max_combined = 0
        max_article = None
        for article in p_active_count.keys():
            if article in i_passive_count.keys():

                iac = i_active_count[article] if article in i_active_count.keys() else 0
                ipc = i_passive_count[article] 
                ppc = p_passive_count[article] if article in p_passive_count.keys() else 0
                pac = p_active_count[article] 

                score = ipc + pac
                if score > max_combined:
                    max_combined = score
                    max_article = article

        print(max_article, p_active_count[max_article], i_passive_count[max_article])
