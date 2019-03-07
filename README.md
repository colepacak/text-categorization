# Text Categorization

## Purpose
I want to categorize news articles using machine learning. To do so, I'm going to:
* tokenize the text using Spacy so that I can clean out the punctuation and stopwords and just have the lemmas (e.g. stems or root word)
* use scikit-learn's CountVectorizer and LinearSVC to test and train a model that can categorize articles
* look for ways to evaluate the model

## Process
Before categorizing the text, I want to get a better idea of what kind of data I'm working with.

First, let's see a breakdown of the category counts throughout the whole dataset to see how evenly represented the categories are - or not.

![Category Counts for BBC Articles](src/figures/category-counts.png)

Next, I'm curious what the most commonly used words are in each category. Right now, I'm just going to use unigrams when tokenizing, but I'm wondering if seeing the most common words might point out the need to use bigrams. Also, it would be interesting to know whether tf-idf may increase accuracy by downscaling the importance of words that appear too often to be useful.

### Most Common Words by category
![Most common words - business](src/figures/most-common-word-counts-business.png)
![Most common words - entertainment](src/figures/most-common-word-counts-entertainment.png)
![Most common words - politics](src/figures/most-common-word-counts-politics.png)
![Most common words - sport](src/figures/most-common-word-counts-sport.png)
![Most common words - tech](src/figures/most-common-word-counts-tech.png)

## TODO
* See how adding a tf-idf transformer to a sklearn pipeline affects accuracy.
* Look at the effect of tf-idf on the top 5 words in each category.

## Local Development
Run `docker-compose up` in the project root to launch a run a Python container that includes all of the dependencies, which can be found in `.docker/requirements.txt`. I downloaded the Spacy model in the dockerfile so that I didn't need to do it manually later on.
