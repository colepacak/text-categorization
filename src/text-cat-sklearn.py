import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import string
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

nlp = spacy.load('en_core_web_sm')
stopwords = stopwords.words('english')
punctuations = string.punctuation
STOPLIST = set(stopwords + list(ENGLISH_STOP_WORDS))
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
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
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

get_most_common_word_counts_by_category()
