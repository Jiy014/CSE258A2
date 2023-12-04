import transformers
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
import torch

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
train_df = train_df.sample(n=10000, ignore_index=True)


# try using BERT final <CLS> embedding
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased").to('cuda:0')

train_text = train_df['review_text'].tolist()
y_train = train_df['rating'].tolist()

embeds = []
batch = 100
with torch.no_grad():
    for i in tqdm(range(len(train_df) // batch)):
        encoded_input = tokenizer(train_text[i*batch:(i+1)* batch], return_tensors='pt', padding=True, truncation=True).to('cuda:0')
        output = model(**encoded_input)
        embeds.append(output.last_hidden_state[:, 0, :].detach().cpu())
        del output

embeds = torch.vstack(embeds)
print('Fitting Model...')
r_model = linear_model.LinearRegression(fit_intercept=True)
r_model.fit(embeds, y_train)
print('Complete!')
## training RMSE: 1.0132697848669168, 10k sample

test_text = test_df['review_text'].tolist()
y_test = test_df['rating'].tolist()
test_embeds = []
batch = 100
with torch.no_grad():
    for i in tqdm(range(len(test_text) // batch)):
        encoded_input = tokenizer(test_text[i*batch:(i+1)* batch], return_tensors='pt', padding=True, truncation=True).to('cuda:0')
        output = model(**encoded_input)
        test_embeds.append(output.last_hidden_state[:, 0, :].detach().cpu())
        del output

test_embeds = torch.vstack(test_embeds)
pred = r_model.predict(test_embeds)
RMSE(y_test, pred)
# 1.0945571351597645







