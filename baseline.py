import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import pandas as pd
import string
import random
from sklearn import linear_model
from sklearn import metrics
from tqdm import tqdm

def RMSE(gold, pred):
    return np.sqrt(np.mean([(a - b) ** 2 for a, b in zip(gold, pred)]))

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        yield d


# load poetry info
poetry_reviews = []
for l in readJSON('goodreads_reviews_poetry.json.gz'):
    poetry_reviews.append(l)

# load comic/graphic info
cg_reviews = []
for l in readJSON('goodreads_reviews_comics_graphic.json.gz'):
    cg_reviews.append(l)

pdf = pd.DataFrame(poetry_reviews)
cdf = pd.DataFrame(cg_reviews)

# prelim investigation stuff
pbids = set(pdf['book_id'].unique())
cbids = set(cdf['book_id'].unique())
# len(pbids.intersection(cbids))
# 54
# len(pbids.union(cbids))
# 125,669

# train test split
# just do 90-10 split over unique reviews
allReviews = pd.concat([pdf, cdf])
allReviews.drop_duplicates(inplace=True, ignore_index=True)
allReviews['is_poetry'] = [allReviews.iloc[i]['book_id'] in pbids for i in range(len(allReviews))]
allReviews['is_comic'] = [allReviews.iloc[i]['book_id'] in cbids for i in range(len(allReviews))]
allReviews = allReviews.sample(frac=1.0, ignore_index = True) # shuffle
train_df = allReviews.iloc[:int(len(allReviews) * 0.9)]
test_df = allReviews.iloc[int(len(allReviews) * 0.9):]
gold = test_df['rating'].tolist() # test set answers

# computer crashing on BoW linear regression
# let's use a subset of 50k training examples
# change as necessary for your computer
train_df = train_df.sample(n=50000, ignore_index=True)

# baseline 0: always predict average
all_avg = train_df['rating'].mean()

def baseline0(test_frame):
    '''
    test_frame: pandas dataframe of reviews
    
    returns: list of predictions for the reviews
    '''
    return [all_avg] * len(test_frame)

pred_b0 = baseline0(test_df)
rmse_b0 = RMSE(gold, pred_b0)
# 1.1960013029069085

# baseline 1: by genre, simply average if multiple genres
poetry_avg = train_df[train_df['is_poetry']]['rating'].mean()
comic_avg  = train_df[train_df['is_comic']]['rating'].mean()

def baseline1(test_frame):
    '''
    test_frame: pandas dataframe of reviews
    
    returns: list of predictions for the reviews
    '''
    pred = []
    for _, row in test_frame.iterrows():
        c = 0
        r = 0
        if row['is_poetry']:
            c += 1
            r += poetry_avg
        if row['is_comic']:
            c += 1
            r += comic_avg
        pred.append(r / c)
    return pred

pred_b1 = baseline1(test_df)
rmse_b1 = RMSE(gold, pred_b1)
# 1.1958966124093255

# baseline 1.5: weight the rating prediction on overlapping genres
#               by frequency of that genre
poetry_count = train_df['is_poetry'].sum()
comic_count = train_df['is_comic'].sum()

def baseline1a(test_frame):
    '''
    test_frame: pandas dataframe of reviews
    
    returns: list of predictions for the reviews
    '''
    pred = []
    for _, row in test_frame.iterrows():
        c = 0
        r = 0
        if row['is_poetry']:
            c += poetry_count
            r += poetry_avg * poetry_count
        if row['is_comic']:
            c += comic_count
            r += comic_avg * comic_count
        pred.append(r / c)
    return pred

pred_b1a = baseline1a(test_df)
rmse_b1a = RMSE(gold, pred_b1a)
# 1.1958993782930691, a little higher, same up to 5 decimal points

# baseline 2: bag of words + linear regression
# preprocessing for bag of words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
### downloading necessary resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
### end download

# build up a dictionary with frequencies
# without stopwords and punctuation.
word_freq = defaultdict(int)
eng_stopwords = stopwords.words('english')
for rev in train_df['review_text']:
    rev = rev.lower()
    ws = [w for w in word_tokenize(rev) if w not in eng_stopwords and w not in string.punctuation]
    for w in ws:
        word_freq[w] += 1

# vocab size: 837989
# prelim investigation stuff:
# counts = [word_freq[k] for k in word_freq]
# max(counts)
# 608129
# np.mean(counts)
# 40.48260538026155
# np.median(counts)
# 1.0
### end investigation

words_sort = sorted([(word_freq[w], w) for w in word_freq], reverse=True)
words_sort[1000] # (4791, 'planet')
words_sort[0]    # (608129, "'s")
words_sort[1]    # (325693, "n't")
# There's only 696891 reviews, so we'll take maybe around 10k words
#   which is a little more than 1% of the words
words_sort[int(len(words_sort) * 0.01)] # (393, 'engrossed'), up to rng
# Choose a truncated vocab of words with frequencies >=: 393
cutoff = words_sort[int(len(words_sort) * 0.01)][0]
vocab = [w for c, w in words_sort if c >= cutoff]
vocab.sort()
wtoi = {w: i for i, w in enumerate(vocab)}
del words_sort
del word_freq

def bow(text):
    '''
    text: a string
    returns: array of size len(vocab) of integers
    '''
    text = text.lower()
    ws = [w for w in word_tokenize(text) if w in wtoi]
    bag = [0] * len(vocab)
    for w in ws:
        bag[wtoi[w]] += 1
    return bag


print('Calculating Training BOWs...')
X_train = [bow(r) for r in tqdm(train_df['review_text'])]
y_train = train_df['rating'].tolist()

print('Fitting Model...')
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
print('Complete!')

print('Calculating Test BOWs...')
X_test = [bow(r) for r in tqdm(test_df['review_text'])]
pred_b2 = model.predict(X_test)
rmse_b2 = RMSE(gold, pred_b2)
# 1.2659206073454594 -> actually higher than before, but maybe because of the subset
# on another subset we get: 1.119605096422715,

# baseline 2a: normalize the counts over the document length
def bow_norm(text):
    '''
    text: a string
    returns: array of size len(vocab) of integers
    '''
    text = text.lower()
    ws = [w for w in word_tokenize(text) if w in wtoi]
    bag = [0] * len(vocab)
    for w in ws:
        bag[wtoi[w]] += 1 / len(ws)
    return bag

print('--Baseline 2a--')
print('Calculating Training BOWs...')
X_train = [bow_norm(r) for r in tqdm(train_df['review_text'])]
y_train = train_df['rating'].tolist()

print('Fitting Model...')
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
print('Complete!')

print('Calculating Test BOWs...')
X_test = [bow_norm(r) for r in tqdm(test_df['review_text'])]
pred_b2a = model.predict(X_test)
rmse_b2a = RMSE(gold, pred_b2a)

'''
On the same subset, we get:

>>> [rmse_b0, rmse_b1, rmse_b1a, rmse_b2, rmse_b2a]
[1.18833455083202, 1.1882608451177044, 1.188261464243391, 1.119605096422715, 1.1170686727052503]
'''
