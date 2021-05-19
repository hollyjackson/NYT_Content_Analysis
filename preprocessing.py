#!/usr/bin/env python3

'''
File name: preprocessing.py
Author: Holly Jackson
Date last modified: 5/19/2021
Python Version: 3.8

This script preprocesses the Concretely Annotated New York Times
and parses articles related to Israel and/or Palestine.

Before running this script, sort and extract all year-month tar.gz
files into two folders, based on the dates of each intifada.

The First Intifada lasted from 12/1987 -- 9/1993.  The Second
Intifada lasted from 9/2000 -- 2/2005.
'''

import os       # standard libraries
import re
import shutil
import argparse

from numba import jit, njit, prange             # for cpu parallelization
from concrete.util import CommunicationReader   # to process the concretely annotated new york times files

class Categorizer:

    def __init__(self):
        # regex commands to clean and separate files
        self.separators = " ", "\t", "\n", "\'s", ".", ",", "-", ";", ":", "(", ")", "[", "]", "/", "\'\'", "\'", "\""
        self.regular_exp = '|'.join(map(re.escape, self.separators))
        # a conservative word bank to tag relevant articles
        self.word_bank = ['Palestine', 'Palestinian', 'Palestinians',
                        'Israel', 'Israeli', 'Israelis']

    def custom_split(self, str_to_split):
        # create regular expression dynamically
        l = re.split(self.regular_exp, str_to_split)
        return list(filter(lambda x: x != '', l))

    @jit(parallel=True)
    def process_period(self, year, month, verbose = False):
        assert type(year) == int and type(month) == int
        
        # year/month combo to string
        if month < 10:
            str_month = '0' + str(month)
        else:
            str_month = str(month)
        year_month = str(year) + str_month

        # count files in directory (not subdirectories)
        path, dirs, files = next(os.walk(intifada + '/' + year_month))
        file_count = len(files)

        # separate out relevant articles
        relevant_articles_israel = []
        relevant_articles_palestine = []
        relevant_articles_both = []

        # parallelize processing 
        for i in prange(file_count):

            article = files[i]
            
            found = False
            comm_file = intifada + '/' + year_month + '/' + article

            # open each concrete file with the concrete-python CommunicationReader            
            for (comm, filename) in CommunicationReader(comm_file):

                if verbose:
                    print("--------------")
                    print("ORIGINAL TEXT")
                    print("--------------")
                    print()
                    print(comm.text)

                # split each text file into component words using regex expression
                words = self.custom_split(comm.text)

                if verbose:
                    print("----------")
                    print("WORD SPLIT")
                    print("----------")
                    print()
                    print(words)

                # identify if it contains any relevant words
                relevance = [word in words for word in self.word_bank]
                if True in relevance[:3]:
                    if verbose:
                        print("FOUND RELEVANT ARICLE -- PALESTINE")
                        print(comm.id)
                    found = True
                    relevant_articles_palestine.append(article)
                if True in relevance[3:]:
                    if verbose:
                        print("FOUND RELEVANT ARICLE -- ISRAEL")
                        print(comm.id)
                    found = True
                    relevant_articles_israel.append(article)
                if True in relevance[:3] and True in relevance[3:]:
                    if verbose:
                        print("FOUND RELEVANT ARICLE -- BOTH")
                        print(comm.id)
                    found = True
                    relevant_articles_both.append(article)

            if verbose:
                print()
                print("-----------------------------------------------")
                print()

        print('For '+str(month) + '/' + str(year) +':')
        print('PALESTINE: '+str(len(relevant_articles_palestine)) + ' relevant articles')
        print('ISRAEL: '+str(len(relevant_articles_israel)) + ' relevant articles')
        print('BOTH: '+str(len(relevant_articles_both)) + ' relevant articles')

        return relevant_articles_palestine, relevant_articles_israel, relevant_articles_both


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--intifada", default='first_intifada', type=str)
    parser.add_argument("--verbose", default=False, type=bool)
    args = parser.parse_args()

    assert args.intifada == 'first_intifada' or args.intifada == 'second_intifada'

    periods = {'first_intifada' : range(1987, 1994), 'second_intifada' : range(2000, 2006) }
    years = periods[args.intifada]
    
    os.makedirs(intifada + '/palestine', exist_ok = True)
    os.makedirs(intifada + '/israel', exist_ok = True)
    os.makedirs(intifada + '/both', exist_ok = True)

    cat = Categorizer()

    total_files_palestine = []
    total_files_israel = []
    total_files_both = []

    for year in years:
        for month in range(1, 13):

            # check if valid path
            if month < 10:
                str_month = '0' + str(month)
            else:
                str_month = str(month)
            year_month = str(year) + str_month

            try:
                path, dirs, files = next(os.walk(intifada + '/' + year_month))
            except:
                print("Cannot find directory "+year_month+ "...continuing to next date")
                continue

            # process this year/month
            relevant_articles_palestine, relevant_articles_israel, relevant_articles_both = cat.process_period(year, month, args.verbose)

            total_files_palestine.append(len(relevant_articles_palestine))
            total_files_israel.append(len(relevant_articles_israel))
            total_files_both.append(len(relevant_articles_both))

            # copy the files to new sorted folders

            print("COPYING FILES")

            os.makedirs(intifada + '/palestine/' + year_month)
            os.makedirs(intifada + '/israel/' + year_month)
            os.makedirs(intifada + '/both/' + year_month)

            for article in relevant_articles_palestine:
                comm_file = intifada + '/' + year_month + '/' + article
                comm_file_dest = intifada + '/palestine/' + year_month + '/' + article
                shutil.copyfile(comm_file, comm_file_dest)

            for article in relevant_articles_israel:
                comm_file = intifada + '/' + year_month + '/' + article
                comm_file_dest = intifada + '/israel/' + year_month + '/' + article
                shutil.copyfile(comm_file, comm_file_dest)

            for article in relevant_articles_both:
                comm_file = intifada + '/' + year_month + '/' + article
                comm_file_dest = intifada + '/both/' + year_month + '/' + article
                shutil.copyfile(comm_file, comm_file_dest)
        
            print("DONE TRANSFERRING FILES")

            print('Successfully processed '+str(month)+'/'+str(year))

    print("Overall, found: ")
    print(str(sum(total_files_palestine))+ ' files related to Palestine')
    print(str(sum(total_files_israel))+ ' files related to Israel')
    print(str(sum(total_files_both))+ ' files related to both')
