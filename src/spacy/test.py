import spacy
import pandas as pd
from utils import clean_text

# Load the model and test data.
nlp = spacy.load('model')
test = pd.read_csv('test.csv')
# Clean and tokenize text.
print("Cleaning the test text...")
test_text = clean_text(test['text'].tolist())

# Test
predictions = []
for doc in nlp.pipe(test_text):
    predictions.append(max(doc.cats, key=doc.cats.get))

test['prediction'] = predictions

# Show metrics
print('Preparing the metrics...\n')
category_len = 15
correct_len = 7
total_len = 7
perc_len = 7
print("{:>{category_len}}  {:>{correct_len}}  {:>{total_len}}  {:>{perc_len}}\n".format(
    '',
    'correct',
    'total',
    'percent',
    category_len=category_len,
    correct_len=correct_len,
    total_len=total_len,
    perc_len=perc_len
))

for category in test['category'].unique():
    df = test[test['category'] == category]
    num_correct = len(df[df['category'] == df['prediction']].index)
    num_total = len(df.index)
    print("{:>{category_len}}  {:>{correct_len}}  {:>{total_len}}  {:>{perc_len},.4f}".format(
        category,
        num_correct,
        num_total,
        num_correct / num_total,
        category_len=category_len,
        correct_len=correct_len,
        total_len=total_len,
        perc_len=perc_len
    ))

num_correct = len(test[test['category'] == test['prediction']].index)
num_total = len(test.index)
print("\n{:>{category_len}}  {:>{correct_len}}  {:>{total_len}}  {:>{perc_len},.4f}\n".format(
    'all',
    num_correct,
    num_total,
    num_correct / num_total,
    category_len=category_len,
    correct_len=correct_len,
    total_len=total_len,
    perc_len=perc_len
))
