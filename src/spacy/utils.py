import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string

nlp = spacy.load('en_core_web_sm')
stopwords = stopwords.words('english')
STOPLIST = set(stopwords + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

# Clean and tokenize text.
def clean_text(docs):
    texts = []
    for doc in docs:
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in STOPLIST]
        tokens = [tok for tok in tokens if tok not in SYMBOLS]
        tokens = ' '.join(tokens)
        texts.append(tokens)

    return texts

# Transform categories to dictionaries.
def get_blank_category_dict(categories):
    dict = {}
    for cat in categories:
        dict[cat] = 0

    return dict

# Transform list of categories to list of category dictionaries.
def transform_categories(categories, data):
    output = []

    for cat in data:
        category_dict = get_blank_category_dict(categories)
        category_dict[cat] = 1
        output.append({ 'cats': category_dict })

    return output
