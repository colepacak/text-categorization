import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import string
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.svm import LinearSVC

nlp = spacy.load('en_core_web_sm')
stopwords = stopwords.words('english')
punctuations = string.punctuation
# Optional list of blacklisted words.
BLACKLISTED_WORDS = ['say']
STOPLIST = set(stopwords + list(ENGLISH_STOP_WORDS))
# Optional list of blacklisted symbols.
BLACKLISTED_SYMBOLS = ["£"]
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

# Load the data.
df = pd.read_csv('data/bbc-text.csv')
# Get the category counts to see how evenly represented each category is in the data.
fig = plt.figure(figsize=(8,4))
sns.barplot(x = df['category'].unique(), y = df['category'].value_counts())
plt.title('Category Counts for BBC Articles')
# plt.savefig('figures/category-counts')

# Get most common words by category.
def clean_text(docs):
    texts = []
    for doc in docs:
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_ for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in STOPLIST]
        tokens = [tok for tok in tokens if tok not in SYMBOLS]
        tokens = ' '.join(tokens)
        texts.append(tokens)

    return texts

def get_most_common_word_counts_by_category():
    for category in df['category'].unique():
        # Get list of articles filtered by category.
        texts = [text for text in df[df['category'] == category]['text']]
        # Get a list cleaned articles.
        texts_clean = clean_text(texts)
        # Get a list of all tokens in the category.
        tokens = ' '.join(texts_clean).split()
        token_counts = Counter(tokens)
        common_words = [word[0] for word in token_counts.most_common(20)]
        common_counts = [word[1] for word in token_counts.most_common(20)]

        fig = plt.figure(figsize=(18,6))
        sns.barplot(x = common_words, y = common_counts)
        plt.title('BBC Articles - Most Common Words - ' + category)
        plt.savefig('figures/most-common-word-counts-' + category)

# get_most_common_word_counts_by_category()

# Split data into train and test.
train, test = train_test_split(df, test_size=0.33, random_state=42)

train_data = train['text'].tolist()
train_labels = train['category'].tolist()

test_data = test['text'].tolist()
test_labels = test['category'].tolist()

def tokenizeText(doc):
    doc = nlp(doc, disable=['parser', 'ner'])
    tokens = [tok.lemma_ for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens

# Multinomial Naive Bayes
pipe_nb = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenizeText)),
    ('clf', MultinomialNB())
])

# Train
pipe_nb.fit(train_data, train_labels)
# Test
predictions_nb = pipe_nb.predict(test_data)

print('Multinomial Naive Bayes:')
print(metrics.classification_report(test_labels, predictions_nb, target_names=df['category'].unique(), digits=5))

# Multinomial Naive Bayes and TF-IDF
pipe_nb_tfidf = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenizeText)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

# Train
pipe_nb_tfidf.fit(train_data, train_labels)
# Test
predictions_nb_tfidf = pipe_nb_tfidf.predict(test_data)

print('Multinomial Naive Bayes with TF-IDF:')
print(metrics.classification_report(test_labels, predictions_nb_tfidf, target_names=df['category'].unique(), digits=5))

# LinearSVC
pipe_svc = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenizeText)),
    ('clf', LinearSVC())
])

# Train
pipe_svc.fit(train_data, train_labels)
# Test
predictions_svc = pipe_svc.predict(test_data)

print('LinearSVC:')
print(metrics.classification_report(test_labels, predictions_svc, target_names=df['category'].unique(), digits=5))

# LinearSVC and TF-IDF
pipe_svc_tfidf = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenizeText)),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC())
])

# Train
pipe_svc_tfidf.fit(train_data, train_labels)
# Test
predictions_svc_tfidf = pipe_svc_tfidf.predict(test_data)

print('LinearSVC with TF-IDF:')
print(metrics.classification_report(test_labels, predictions_svc_tfidf, target_names=df['category'].unique(), digits=5))
