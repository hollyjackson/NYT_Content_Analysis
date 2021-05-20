# Content Analysis of the *New York Times*

This code accompanies the pre-print ["The *New York Times* Distorts the Palestinian Struggle: A Case Study of Anti-Palestinian Bias in American News Coverage of the First and Second Palestinian Intifadas"](http://web.mit.edu/hjackson/www/The_NYT_Distorts_the_Palestinian_Struggle.pdf), recently submitted to the *Journal of Palestine Studies*.  The study proves a history of bias against Palestine in the *New York Times* during the First and Second Palestinian Intifadas.

This codebase is run on an archived version of the concretely-annotated *New York Times*, documented at this reference:

> Francis Ferraro, Max Thomas, Matthew Gormley, Travis Wolfe, Craig Harman, and Benjamin Van Durme. "Concretely Annotated Corpora." *In The Proceedings of the NIPS Workshop on Automated Knowledge Base Construction (AKBC)*. NIPS Workshop 2014.

My content analysis was performed in python 3.8 on a 16-core Ubuntu 18.04 machine

## Setup

Clone the repo and submodules

```shell
git clone --recurse-submodules https://github.com/hollyjackson/NYT_Content_Analysis.git
```

### Requirements

This codebase requires a number of natural language processing libraries:

* [spaCy](https://spacy.io/)
* [NLTK](https://www.nltk.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [Keras](https://keras.io/)

These, along with other required libraries, can all be installed as follows:

```shell
pip3 install -r requirements.txt
```

## Usage

### `preprocessing.py`

This script preprocesses the Concretely Annotated *New York Times* and parses articles related to Israel and/or Palestine in a specific year of a specific Intifada.  Before running this script, sort and extract all year-month `tar.gz` files into two folders, based on the dates of each Intifada.

The script can be run as follows with two optional command-line arguments:

```shell
python3 preprocessing.py --intifada {'first_intifada' or 'second_intifada'} --verbose {True or False}
```

The defaults are `'first_intifada'` and `False`.

### `voice_identifier.py`

This script processes articles related to Israel and/or Palestine in the Concretely Annotated *New York Times* and classifies the voice (active or passive) of all sentences with Israeli or Palestinian subjects.  It should be run after `preprocessing.py` (all pre-processed data must be stored in a subdirectory called `sorted_files`).

The script can be run as follows with two required command-line arguments:

```shell
python3 voice_identifier.py --intifada {'first_intifada' or 'second_intifada'} --year {must be a valid year in the respective Intifada}
```

This can easily be parallelized using a `bash` scrhipt.

### `voice_count.py`

This script instances of passive and active voice in articles processed by my NLP pipeline.  It should be run after `preprocessing.py` and `voice_identifier.py`.

### `generate_training_set.py`

This script provides a user interface to generate a training set from a sample of 500 random articles on Palestine and/or Israel from the *New York Times* during the First and Second Intifadas.  It should be run after `preprocessing.py`.  When run, the user will be prompted to blindly tag the violent sentiment of the set of words appearing in this sample.  The results will be saved to `training_data.csv`.

### `violence_count.py`

This script counts references to violence in articles processed by my NLP pipeline.  It should be run after `preprocessing.py` and `generate_training_set.py`.

### Plotting

I have not included any plotting scripts since they were specific to the the data I mined and the topics I focused on.  There are many equally successful and useful ways to plot the data, and those should be explored by the researcher. 

