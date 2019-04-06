import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from utils import transform_categories, clean_text
import random

nlp = spacy.load('en_core_web_sm')

# Load data.
df = pd.read_csv('../data/bbc-text.csv')

train, test = train_test_split(df, test_size=0.33, random_state=42)
print("Cleaning the training text...")
train_text = clean_text(train['text'].tolist())

# Export test data to CSV for later use when testing.
test.to_csv('test.csv', index=False)

# Transform categories to dictionaries.
print("Preparing the training labels...")
categories = df['category'].unique()
train_labels = transform_categories(categories, train['category'].tolist())

# Train
print("Training the model...")
textcat = nlp.create_pipe('textcat')
nlp.add_pipe(textcat, last=True)

for cat in categories:
    textcat.add_label(cat)

optimizer = nlp.begin_training()

TRAIN_DATA = list(zip(train_text, train_labels))

for i in range(5):
    random.shuffle(TRAIN_DATA)
    for text, category in TRAIN_DATA:
        doc = nlp.make_doc(text)
        nlp.update([doc], [category], sgd=optimizer)

print("Saving the model to disk...")
nlp.to_disk('model')
