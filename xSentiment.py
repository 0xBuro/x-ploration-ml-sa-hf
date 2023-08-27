import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

# read csv file 
# kaggle data src: (https://www.kaggle.com/datasets/tleonel/crypto-tweets-80k-in-eng-aug-2022)
csv_file_path = "crypto-query-tweets.csv"
df = pd.read_csv(csv_file_path, usecols=['date_time', 'username', 'verified', 'tweet_text'])

# filter for verified users only
filtered_df = df[df['verified'] == True]


# preprocess text method 
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# use sentiment task for base roberta
task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# autotokenizer hf
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# label mapping from tweetEval with sentiment as task
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# load the Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)


# get sentiment scores
def get_sentiment(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    return scores

# analyze sentiment for every row, return labels and score in sentiment_result array
def analyze_sentiment(row):
    scores = get_sentiment(row['tweet_text'])
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    sentiment_results = []
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        sentiment_results.append((l, np.round(float(s), 4)))
    return sentiment_results

# Expand column to sentiment, apply sentiment analyze to its rows
filtered_df['sentiment'] = filtered_df.apply(analyze_sentiment, axis=1)

# print new dataframe
print(filtered_df)