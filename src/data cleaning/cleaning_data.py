'''
Created on 8 Feb 2020

@author: puteri ika
'''
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download()

#get data
review_df= pd.read_csv('Raw Data.csv')
print(review_df.shape)
print(review_df.head(2))
review_df['Cust_Reviews_raw']=review_df['Cust_Reviews']
import re

review_df['Cust_Reviews']=review_df['Cust_Reviews'].tolist()
def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

review_df['Cust_Reviews'] = review_df['Cust_Reviews'].apply(lambda x: remove_punct(x))
p=review_df['Cust_Reviews'] 
review_df['punctuation']=p
review_df['Cust_Reviews']=review_df['Cust_Reviews'].str.split("//",n=4,expand=True)
#tokenization
#(r'\w') pattern of regEx
tokenizer= RegexpTokenizer(r'\w+')
review_df['Cust_Reviews']= review_df['Cust_Reviews'].apply(lambda x:tokenizer.tokenize(x.lower()))
token=review_df['Cust_Reviews']
print("Tokenization and Lowercase")
print(review_df['Cust_Reviews'].head(20))
token=pd.DataFrame(review_df,columns=['Cust_Reviews'])
review_df['token']=token
tk=pd.DataFrame(review_df,columns=['Cust_Reviews'])
tk.to_csv('token.csv') 

#stopword 
def remove_stopword(text):
    words=[w for w in text if w not in stopwords.words('english')]
    return words

review_df['Cust_Reviews']= review_df['Cust_Reviews'].apply(lambda x:remove_stopword(x))
s=review_df['Cust_Reviews']
print("Stopword")
review_df['stopwords']=s
print(review_df['Cust_Reviews'].head(20))
sw=pd.DataFrame(review_df,columns=['Cust_Reviews'])


#lemmatization
lemmatizer=WordNetLemmatizer()

def word_lemmatizer(text):
    lem_sent=' '.join([lemmatizer.lemmatize(i, pos="v") for i in text])
    return lem_sent
#provide the context in which you want to lemmatize that is the parts-of-speech (POS). This is done by giving the value for pos parameter in wordnet_lemmatizer.lemmatize.
review_df['Cust_Reviews']=review_df['Cust_Reviews'].apply(lambda x: word_lemmatizer(x))       
print("lemmatization")
l=review_df['Cust_Reviews']
review_df['lemma']=l
print(review_df['Cust_Reviews'].head(20))

labeled=pd.DataFrame(review_df,columns=['Cust_Reviews','Brands',''])
labeled.to_csv('lemma.csv') 

'''
print("{0:20}{1:20}".format("Word","Lemma"))
sentence_words =review_df['Cust_Reviews'].tolist()
for word in range(sentence_words):
    print ("{0:20}{1:20}".format(word,lemmatizer.lemmatize(word)))
lemma=pd.DataFrame(review_df,columns=['Cust_Reviews'])
lemma.to_csv('lemmatization.csv') 
'''

sid = SentimentIntensityAnalyzer()

def sentiment_score(x):
    print(x)
    scores = sid.polarity_scores(x)
    for key in sorted(scores):
        print('{0}:{1}, '.format(key, scores[key]), end='')
        print()
    return scores
             
review_df['sentiment'] = review_df['Cust_Reviews'].apply(lambda x:sentiment_score(x))
print(review_df['sentiment'].head(2))
        
def convert(x):
    if x < 0.05:
        return "negative"
    elif x > 0.05:
        return "positive"  
review_df['Sentiment_label'] = review_df['sentiment'].apply(lambda x:convert(x['compound']))

labeled=pd.DataFrame(review_df,columns=['Cust_Reviews_raw','Brands','punctuation','token','stopwords','lemma','Sentiment_label'])
labeled.to_csv('label.csv') 

from wordcloud import WordCloud
import matplotlib.pyplot as plt


neg = review_df[review_df.Sentiment_label == 'negative']
neg_r = []
for t in neg.Cust_Reviews:
    neg_r.append(t)
neg_r = pd.Series(neg_r).str.cat(sep=' ')
review_df['negative']=neg_r

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_r)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

pos = review_df[review_df.Sentiment_label == 'positive']
pos_r = []
for t in pos.Cust_Reviews:
    pos_r.append(t)
pos_r = pd.Series(pos_r).str.cat(sep=' ')
review_df['positive']=pos_r

wordc = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(pos_r) 
plt.figure(figsize=(12,10))
plt.imshow(wordc, interpolation="bilinear")
plt.axis("off")
plt.show()

tokenizer= RegexpTokenizer(r'\w+')
review_df['Cust_Reviews']= review_df['Cust_Reviews'].apply(lambda x:tokenizer.tokenize(x.lower()))
token=review_df['Cust_Reviews']
#print(lemmatizer.lemmatize('going', wordnet.VERB))




