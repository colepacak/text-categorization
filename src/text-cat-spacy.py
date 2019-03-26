import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string

nlp = spacy.load('en_core_web_sm')
stopwords = stopwords.words('english')
STOPLIST = set(stopwords + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

# Load data
df = pd.read_csv('data/bbc-text.csv')

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

texts = df['text'].tolist()
train_texts = clean_text(texts)

# Transform categories to dictionary.
categories = df['category'].unique()
train_categories = []

def get_blank_category_dict(categories):
    dict = {}
    for cat in categories:
        dict[cat] = 0

    return dict

for cat in df['category'].tolist():
    category_dict = get_blank_category_dict(categories)
    category_dict[cat] = 1
    train_categories.append({ 'cats': category_dict })

# Put training data back together.
train_data = list(zip(train_texts, train_categories))

# Train
textcat = nlp.create_pipe('textcat')
nlp.add_pipe(textcat, last=True)

for cat in categories:
    textcat.add_label(cat)

optimizer = nlp.begin_training()

for text, category in train_data:
    doc = nlp.make_doc(text)
    nlp.update([doc], [category], sgd=optimizer)

test = "Almost three years ago the UK voted in favour of leaving the EU. Yet the original departure date of 29 March has been delayed and the government is searching for a way forward. So, what do the majority of the UK's voters now think about Brexit? How well have talks been handled? Theresa May argues that her deal is the best way of fulfilling the instruction to leave given by voters in the EU referendum. Trouble is, voters themselves - including not least those who voted Leave - have become deeply critical of how the UK government has handled Brexit negotiations. This is seen in data from a newly published survey from NatCen Social Research, conducted between 24 January and 17 February. Two years ago, those who voted Leave were inclined to give the government the benefit of the doubt. However, as many as 80% of Leave voters now say that it has handled Brexit negotiations badly. That figure is almost as high as it is among Remain voters (85%), who had previously been more critical of the government's approach. Remarkably, Leave voters are now just as critical of the UK government's role as they are of the EU's: 79% of Leave supporters say the EU has handled Brexit badly. Will the UK get a good deal? Meanwhile, the longer negotiations have continued, the more pessimistic voters have become about how good a deal the UK will secure. Two years ago, there were almost as many who thought that the UK would obtain a good deal (33%) as thought it would find itself with a bad one (37%). However, that mood soon changed and by last summer as many as 57% reckoned the UK would emerge with a bad deal. Now that the first phase of the Brexit negotiations has been concluded - though, as yet at least, not approved by MPs - the proportion who think the UK is heading for a bad deal is, at 63%, even higher. Here too, Remain and Leave voters are now largely in agreement. This is despite Leave voters initially being much more positive. As many as 66% of Leave supporters now believe that the UK is faced with a bad deal - even more than the 64% of Remain voters who express that view. It seems that the prime minister's deal has failed to satisfy many of the very voters whose wishes the deal is intended to fulfil. Do UK voters still want to leave the EU? But does this negative reaction to the deal mean voters have changed their minds about leaving the EU in the first place? In truth, a moving average of the six most recent polls, has been indicating for some time that slightly more people now say they would vote Remain than Leave in another ballot. At present, the average level of support for the two options (after Don't Knows are excluded) is Remain 54%, Leave 46%. In part, this is because Leave voters are a little less likely to say they would vote the same way again (82%), than Remain voters are (86%). But the swing to Remain, such as it is, is also down to those who did not vote in 2016. Of this group, 43% say they would vote Remain, whereas 19% say they would back Leave. In truth, the polls are too close for opponents of Brexit to assume that a second ballot would produce a different result. But, equally, supporters of Brexit cannot say with confidence that the balance of opinion remains as it was in June 2016. That depends on how they are asked. Some polls introduce the idea of another ballot as a people's vote, or a public vote and do not make it clear that remaining in the EU would be an option. When put to the public in this way, the polls suggest on average that supporters of a second referendum outnumber opponents by 12 points. However, the average level of support and opposition changes when people are asked if there should be another referendum with Remain as an option on the ballot paper. Asked in this way, opponents outnumber supporters by nine points. But perhaps what is more important is that Remain voters are much keener on this idea than Leave supporters. That suggests the proposal is not yet a way out of the Brexit impasse that is backed by both sides in the Brexit debate. Remain and Leave voters may agree that they do not like Mrs May's deal, but that does not mean that they agree on what should happen instead."

doc1 = nlp(test)

print(doc1.cats)
# {'tech': 0.002452729269862175, 'business': 0.6330711245536804, 'sport': 0.03160744905471802, 'entertainment': 0.0004235804080963135, 'politics': 0.8862512111663818}
