'''
Created on 14 Mar 2020

@author: User
'''
'''
Created on 9 Feb 2020

@author: User
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
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.layers import Dense, Input, Flatten, Dropout, Add
from keras.models import Model
from keras.layers.merge import concatenate, add
import gensim
from keras import backend
from keras.layers.embeddings import Embedding
from gensim.models import Word2Vec
import os
import re
from keras.models import Sequential
#nltk.download()

#get data

#review_df= pd.read_csv('review1.csv',encoding='ISO-8859-1')
review_df= pd.read_csv('Mobile_Brand_s.csv')
print(review_df.shape)
print(review_df.head(2))


def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct
review_df['Cust_Reviews'] = review_df['Cust_Reviews'].apply(lambda x: remove_punct(x))

review_df['Cust_Reviews']=review_df['Cust_Reviews'].str.split("//",n=4,expand=True)

#tokenization
#(r'\w') pattern of regEx
tokenizer= RegexpTokenizer(r'\w+')
review_df['Cust_Reviews']= review_df['Cust_Reviews'].apply(lambda x:tokenizer.tokenize(x.lower()))
token=review_df['Cust_Reviews']
print("Tokenization and Lowercase")
print(review_df['Cust_Reviews'].head(5))
'''
#stopword 
def remove_stopword(text):
    words=[w for w in text if w not in stopwords.words('english')]
    return words

review_df['Cust_Reviews']= review_df['Cust_Reviews'].apply(lambda x:remove_stopword(x))
print("Stopword")
print(review_df['Cust_Reviews'].head(5))
'''
#lemmatization
lemmatizer=WordNetLemmatizer()

def word_lemmatizer(text):
    lem_sent=" ".join([lemmatizer.lemmatize(i,pos="v") for i in text])
    return lem_sent

review_df['Cust_Reviews']=review_df['Cust_Reviews'].apply(lambda x: word_lemmatizer(x))
print("lemmatization")
print(review_df['Cust_Reviews'].head(5))

pos = []
neg = []

for l in review_df.Polarity:
    if l == 0:
        pos.append(0)
        neg.append(1)
        
    elif l == 1:
        pos.append(1)
        neg.append(0)
        
   
        
review_df['Positive']= pos
review_df['Negative']= neg

print(review_df[['Cust_Reviews','Brands','Polarity','Positive','Negative']].head(2))
labeled=pd.DataFrame(review_df,columns=['Cust_Reviews','Brands','Polarity','Positive','Negative'])
labeled.to_csv('clean_data_visualise.csv') 
#print(lemmatizer.lemmatize('going', wordnet.VERB))

#Clean data
#print(review_df[['Cust_Reviews','Polarity']].head(5))
'''
pd.set_option('display.max_rows', review_df.shape[0]+1)
print(review_df)

'''
tokenizer= RegexpTokenizer(r'\w+')
review_df['token']= review_df['Cust_Reviews'].apply(lambda x:tokenizer.tokenize(x.lower()))

#data splitting

data_train, data_test = train_test_split(review_df, test_size=0.10, random_state=42)

#training

all_training_words = [word for tokens in data_train['token'] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train['token']]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("Training Words:-")
print()
print("%s total of words , with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Maximum sentence length is %s" % max(training_sentence_lengths))

#testing
all_test_words = [word for tokens in data_test['token'] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test['token']]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("Testing Words :-")
print()
print("%s total of words, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Maximum sentence length is %s" % max(test_sentence_lengths))

#wors2vec for embedding word
word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['token'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)


training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)
'''Tokenize and Pad sequences
Each word is assigned an integer and that integer is placed in a list. As all the training sentences must have same input shape we pad the sentences.
'''
EMBEDDING_DIM = 300 # how big is each word vector
MAX_VOCAB_SIZE = 5000 # how many unique words to use (i.e num rows in embedding vector)
MAX_SEQUENCE_LENGTH = 500 #cut off reviews after 100 words

tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train['Cust_Reviews'].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train['Cust_Reviews'].tolist())
train_word_index = tokenizer.word_index
train_cnn_data = pad_sequences(training_sequences, 
                               maxlen=MAX_SEQUENCE_LENGTH)
train_embedding_weights = np.zeros((len(train_word_index)+1, 
 EMBEDDING_DIM))


print("Found %s unique tokens." % len(train_word_index))
print(train_cnn_data)

for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
 
print(train_embedding_weights.shape)
print(train_embedding_weights)

test_sequences = tokenizer.texts_to_sequences(data_test['Cust_Reviews'].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
 
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)#not trained
    my_embeddings = embedding_layer.get_weights()
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    filter_sizes = [2,3,4,5,6]
    #filter_sizes = [8,9,10,11,12]
    #Convolution1D, which will learn filters
    for filter_size in filter_sizes:
        layer_conv = Conv1D(filters=200, 
                        kernel_size=filter_size, 
                        activation='relu')(embedded_sequences)
        #layer_pool= MaxPooling1D(5)(layer_conv)
        layer_pool = GlobalMaxPooling1D()(layer_conv)
        convs.append(layer_pool)
    layer_merge = concatenate(convs, axis=1)
    x = Dropout(0.1)(layer_merge)  
    #x=Flatten()(x)
    x = Dense(128, activation='relu')(x) #fully_connected layer
    x = Dropout(0.1)(x)
    softmax_layer_probability = Dense(labels_index, activation='softmax')(x) # sigmoid not suitable for relu activation.. it will blow up the relu activation
    model = Model(sequence_input, softmax_layer_probability)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
                  
    model.summary()
    return model

label_names = ['Negative','Positive']

#OUTPUT LABELS
y_train = data_train[label_names].values
y_test=data_test[label_names].values

x_train = train_cnn_data
y_tr = y_train

model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                len(list(label_names)))
'''
from sklearn.utils import class_weight

class_weight = {0: 1.,
                1: 50.,
                2: 2.}
'''
num_epochs = 6
batch_size = 32 #based on memory capability
History= model.fit(train_cnn_data, 
                 y_tr, 
                 epochs=num_epochs, 
                 validation_split=0.1, 
                 shuffle=True, 
                 batch_size=batch_size)#, class_weight=class_weight


import matplotlib.pyplot as plt

accuracy = History.history['acc']
val_accuracy = History.history['val_acc']
loss =History.history['loss']
val_loss = History.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



#testing
testing_predictions = model.predict(test_cnn_data, 
                            batch_size=32, 
                            verbose=1)#1024
labels = [0, 1]

prediction_labels=[]
for p in testing_predictions:
    prediction_labels.append(labels[np.argmax(p)])   
#print("prediction")
print(sum(data_test.Polarity==prediction_labels)/len(prediction_labels))
print(data_test.Polarity.value_counts())

scores = model.evaluate(test_cnn_data, y_test, verbose=1)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(data_test.Polarity, prediction_labels)
print("confusion matrix : ")
print(cm)

# Accuracy
from sklearn.metrics import accuracy_score
acc=accuracy_score(data_test.Polarity, prediction_labels)
print("Accuracy : ")
print(acc)
# Recall
from sklearn.metrics import recall_score
recall=recall_score(data_test.Polarity,prediction_labels, average=None)
print("recall : ")
print(recall)
# Precision
from sklearn.metrics import precision_score
precision=precision_score(data_test.Polarity, prediction_labels, average=None)
print("precision : ")
print(precision)

# Method 1: sklearn
from sklearn.metrics import f1_score
f1_score(data_test.Polarity, prediction_labels, average=None)
# Method 2: Manual Calculation
F1 = 2 * (precision * recall) / (precision + recall)
print("F1 score : ")
print(cm)
# Method 3: Classification report [BONUS]
from sklearn.metrics import classification_report
print(classification_report(data_test.Polarity,prediction_labels, target_names=label_names))

print("Training accuracy :")
print(History.history['acc'])

print("Testing :")
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('CNN_engine_model_sentiment_analysis.h5')



max_len=500
print("New Data by using CNN classifier: ")
print()
raw_text="worst phone that i had!"

print(raw_text)

review_df['New Data']=raw_text
seq= tokenizer.texts_to_sequences([raw_text])
text_padded = pad_sequences(seq, maxlen=max_len)
predict_sentences= model.predict_on_batch(text_padded)

#print(predict_sentences)
print(label_names[np.argmax(predict_sentences)])

predict_sentences=np.argmax(predict_sentences)
#print(predict_sentences)

max_len=500
raw_text="terrible, worst, bad, not supported phone ever, the camera is soo good!"
review_df['New Data']=raw_text
print(raw_text)

seq= tokenizer.texts_to_sequences([raw_text])
text_padded = pad_sequences(seq, maxlen=max_len)
predict_sentences= model.predict(text_padded)

#print(predict_sentences)
print(label_names[np.argmax(predict_sentences)])

predict_sentences=np.argmax(predict_sentences)

#print(predict_sentences)


max_len=500
raw_text="beautiful and best phone ever!"
review_df['New Data']=raw_text
print(raw_text)

seq= tokenizer.texts_to_sequences([raw_text])
text_padded = pad_sequences(seq, maxlen=max_len)
predict_sentences= model.predict_on_batch(text_padded)

#print(predict_sentences)
print(label_names[np.argmax(predict_sentences)])

predict_sentences=np.argmax(predict_sentences)

#print(predict_sentences)


max_len=500
raw_text="this phone not supported around this area!"
review_df['New Data']=raw_text
print(raw_text)

seq= tokenizer.texts_to_sequences([raw_text])
text_padded = pad_sequences(seq, maxlen=max_len)
predict_sentences= model.predict_on_batch(text_padded)

#print(predict_sentences)
print(label_names[np.argmax(predict_sentences)])

predict_sentences=np.argmax(predict_sentences)

#print(predict_sentences)
max_len=500
raw_text="this phone one of my favourite because of their feature like camera, 4G connection.like it so much"
review_df['New Data']=raw_text
print(raw_text)

seq= tokenizer.texts_to_sequences([raw_text])
text_padded = pad_sequences(seq, maxlen=max_len)
predict_sentences= model.predict_on_batch(text_padded)

#print(predict_sentences)
print(label_names[np.argmax(predict_sentences)])

predict_sentences=np.argmax(predict_sentences)


max_len=500
raw_text="this phone not beautiful at all but i like it!"
review_df['New Data']=raw_text
print(raw_text)

seq= tokenizer.texts_to_sequences([raw_text])
text_padded = pad_sequences(seq, maxlen=max_len)
predict_sentences= model.predict_on_batch(text_padded)

#print(predict_sentences)
print(label_names[np.argmax(predict_sentences)])

predict_sentences=np.argmax(predict_sentences)

#print(predict_sentences)


    
max_len=500
raw_text="i received the phone,i like it sooo much. it is not sound properly and the camera is worst, what a terrible phone!"
review_df['New Data']=raw_text
print(raw_text)

seq= tokenizer.texts_to_sequences([raw_text])
text_padded = pad_sequences(seq, maxlen=max_len)
predict_sentences= model.predict_on_batch(text_padded)

#print(predict_sentences)
print(label_names[np.argmax(predict_sentences)])

predict_sentences=np.argmax(predict_sentences)

#print(predict_sentences)

    
max_len=500
raw_text="this phone is fake, what a bad seller!"
review_df['New Data']=raw_text
print(raw_text)

seq= tokenizer.texts_to_sequences([raw_text])
text_padded = pad_sequences(seq, maxlen=max_len)
predict_sentences= model.predict_on_batch(text_padded)

#print(predict_sentences)
print(label_names[np.argmax(predict_sentences)])

predict_sentences=np.argmax(predict_sentences)

#print(predict_sentences)
nd=pd.read_csv('iphone.csv')#,encoding='ISO-8859-1'

def new_data(x):
    max_len=500
    raw_text=x
    seq= tokenizer.texts_to_sequences([raw_text])
    text_padded = pad_sequences(seq, maxlen=max_len)
    predict_sentences= model.predict_on_batch(text_padded)
    r=label_names[np.argmax(predict_sentences)]
    return r

nd['sentiment_label']= nd['Cust_Reviews'].apply(lambda x:new_data(x))
labeled=pd.DataFrame(nd,columns=['Cust_Reviews','sentiment_label'])
labeled.to_csv('new_data_iphone.csv')  

#print(predict_sentences)
if predict_sentences == 0:
    review_df['result']= "Negative" 
else:
    review_df['result']= "Positive"
    
labeled=pd.DataFrame(review_df,columns=['New Data','result'])
labeled.to_csv('result_new_data.csv')    
#SAVE MODEL

'''
model_json = model.to_json()
with open("model_json.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_json.h5")
print("Saved model to disk")
'''
from tkinter import*
from tkinter import messagebox
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize 
from keras.models import Model
from keras.utils import np_utils
import tkinter as tk
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from tkinter.filedialog import asksaveasfilename
import os
from tkinter import Menu

label_names = ['Negative','Positive']

class sentiment_analysis:
    
    def next(self):
        top=Toplevel()
        menu=Menu(top)
        new_item = Menu(menu)
        new_item.add_command(label="Exit", command=root.quit)
        menu.add_cascade(label='File', menu=new_item)
        
        helpmenu = Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=self.About2)
        top.config(menu=menu)
        
        top.title("Sentiment Analyzer Application")
        top.geometry("900x600+0+0")
        
        self.bg_icon=ImageTk.PhotoImage(file="samsung9_1553742024.jpg")
        self.ufile=StringVar()
        self.savefile=StringVar()
        
        bg_lbl=Label(top,image=self.bg_icon).pack()
        title=Label(top,text="Sentiment Analyzer for Mobile Brands",font=("Countryside",30,"bold"),bg="pink",fg="white",bd=10,relief=GROOVE)
        title.place(x=0,y=0,relwidth=1)
        
        file_upload_frame=tk.Frame(top, width=100, height=700, background="palevioletred")
        file_upload_frame.place(x=250,y=150)
        self.txtlbl=Entry(file_upload_frame,width=30,bd=10,textvariable=self.ufile,relief=GROOVE,font=("",12)).grid(row=1,column=1,padx=10,pady=20)
        btn_upload=Button(file_upload_frame,text="Select File",width=13,command=self.Upload,font=("times new roman",12,"bold"),bg="black",fg="white").grid(row=1,column=2,padx=5,pady=5)
        btn_upload=Button(file_upload_frame,text="Analyze",width=15,command=self.analyze,font=("times new roman",12,"bold"),bg="black",fg="white").grid(row=3,column=1,padx=50,pady=10)
        self.txtlbl2=Entry(file_upload_frame,width=30,bd=5,textvariable=self.savefile,relief=GROOVE,font=("",12)).grid(row=4,column=1,padx=10,pady=10)
        btn_upload=Button(file_upload_frame,text="Bar chart",width=15,command=self.bar,font=("times new roman",12,"bold"),bg="black",fg="white").grid(row=5,column=1,padx=50,pady=10)
        btn_upload=Button(file_upload_frame,text="Wordcloud Positive",width=15,command=self.wordcloudpositive,font=("times new roman",12,"bold"),bg="black",fg="white").grid(row=6,column=1,padx=50,pady=10)
        btn_upload=Button(file_upload_frame,text="Wordcloud Negative",width=15,command=self.wordcloudnegative,font=("times new roman",12,"bold"),bg="black",fg="white").grid(row=6,column=2,padx=50,pady=10)
        btn_upload=Button(file_upload_frame,text="Bar positive",width=15,command=self.positive,font=("times new roman",12,"bold"),bg="black",fg="white").grid(row=7,column=1,padx=50,pady=10)
        btn_upload=Button(file_upload_frame,text="Bar negative",width=15,command=self.negative,font=("times new roman",12,"bold"),bg="black",fg="white").grid(row=7,column=2,padx=50,pady=10)
       
        windowWidth = top.winfo_reqwidth()
        windowHeight = top.winfo_reqheight()
        print("Width",windowWidth,"Height",windowHeight)
 
# Gets both half the screen width/height and window width/height
        positionRight = int(top.winfo_screenwidth()/4 - windowWidth/4)
        positionDown = int(top.winfo_screenheight()/10 - windowHeight/10)
 
# Positions the window in the center of the page.
        top.geometry("+{}+{}".format(positionRight, positionDown))
        
        
    def textNew(self):
        
        self.resultn.set("")
        self.resultp.set("")
        
        if self.utext.get()=="":
            messagebox.showerror("error", "please enter your text!")
        else:
            raw_text=self.utext.get()
            
            #model=load_model('CNN_engine_model_sentiment_analysis.h5')
            
            #dataset = pd.read_csv("movie.csv",encoding='ISO-8859-1')
            # split into input (X) and output (Y) variables
            #data_train, data_test = train_test_split(dataset, test_size=0.10, random_state=42)
            # evaluate the model
            #score = model.evaluate(data_train, data_test, verbose=0)
            #print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
            
            max_len=500
            review_df['New Data']=raw_text
            print(raw_text)

            seq= tokenizer.texts_to_sequences([raw_text])
            text_padded = pad_sequences(seq, maxlen=max_len)
            predict_sentences= model.predict_on_batch(text_padded)
            
            print(predict_sentences)
            print(label_names[np.argmax(predict_sentences)])
            print(predict_sentences)
            x="This is %s sentiment with %2f%% confidence" % (label_names[np.argmax(predict_sentences)], predict_sentences[0][np.argmax(predict_sentences)] * 100)
            
            self.confident.set(x)
            predict_sentences=np.argmax(predict_sentences)
 
            #print(classification_report(np.argmax(predict_sentences),y_pred=predict_sentences,target_names=label_names))

            if predict_sentences == 0:
                self.resultn.set("Negative")
               
            else:
                self.resultp.set("Positive")
                
    def About(self):
        top=Toplevel()
        top.title("About Sentiment Analysis Application")
        top.geometry("900x120+0+0")
        string1 ='This sentiment analyzer is to analyze the sentiment whether it is positive or negative'+'\n'+'Next page button is to the next section of this application.'+'\n'+''+'At the next section it is provide function to analyse your reviews from the customers'+'\n'+' and there are buttons for data visualisation'
        title=Label(top,text=string1,font=("Quite Magical",18,"bold"),bg="pink",fg="white",bd=10)
        title.place(x=0,y=0,relwidth=1)
        
        windowWidth = top.winfo_reqwidth()
        windowHeight = top.winfo_reqheight()
        print("Width",windowWidth,"Height",windowHeight)
        positionRight = int(top.winfo_screenwidth()/4 - windowWidth/4)
        positionDown = int(top.winfo_screenheight()/2 - windowHeight/2)
 
# Positions the window in the center of the page.
        top.geometry("+{}+{}".format(positionRight, positionDown))
    
    def About2(self):
        top=Toplevel()
        top.title("About Sentiment Analysis Application")
        top.geometry("900x240+0+0")
        string1 ='This sentiment analyzer is to analyze the sentiment whether it is positive or negative'+'\n'+'Select file button will allow you to select the .csv file.'+'\n'+'(note: this application only accept csv file for the analysis)'+'\n'+'Please make sure attribute name for reviews is Cust_Reviews and for brands name as Brands'+'\n'+'bar chart button will shows the count of positive and negative of the reviews data'+'\n'+'Wordcloud positive and negative button will representing text data in'+'\n'+'which the size of each word indicates its frequency or importance'+'\n'+'the bigger size is the most mentioned in the sentences'+'\n'+'Bar positive and negative button will show the count of positive and negative reviews for each Brands'
        title=Label(top,text=string1,font=("Quite Magical",18,"bold"),bg="pink",fg="white",bd=10)
        title.place(x=0,y=0,relwidth=1)
        
        windowWidth = top.winfo_reqwidth()
        windowHeight = top.winfo_reqheight()
        print("Width",windowWidth,"Height",windowHeight)
        positionRight = int(top.winfo_screenwidth()/4 - windowWidth/4)
        positionDown = int(top.winfo_screenheight()/2 - windowHeight/2)
 
# Positions the window in the center of the page.
        top.geometry("+{}+{}".format(positionRight, positionDown))
            
                
    def __init__(self,root):
        
        self.root=root
        
        menu=Menu(self.root)
        new_item = Menu(menu)
        new_item.add_command(label="Exit", command=root.quit)
        menu.add_cascade(label='File', menu=new_item)
        
        helpmenu = Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=self.About)
        self.root.config(menu=menu)
        
        self.root.title("Sentiment Analysis Application")
        self.root.geometry("900x600+0+0")
        self.root.bg_icon=ImageTk.PhotoImage(file="samsung9_1553742024.jpg")
        
        self.utext=StringVar()
        self.resultp=StringVar()
        self.resultn=StringVar()
        self.confident=StringVar()
        
        bg_lbl=Label(self.root,image=self.root.bg_icon).pack()
        title=Label(self.root,text="Sentiment Analyzer for Mobile Brands",font=("Countryside",30,"bold"),bg="pink",fg="white",bd=10,relief=GROOVE)
        title.place(x=0,y=0,relwidth=1)
        
        file_upload_frame=Frame(self.root,bg="palevioletred",highlightbackground="pink",height=400,padx=20,pady=10,borderwidth=3)
        file_upload_frame.place(x=150,y=150)
        
        txtlbl=Entry(file_upload_frame,width=30,bd=5,textvariable=self.utext,relief=GROOVE,font=("",15)).grid(row=1,column=1,padx=10,pady=20,ipadx=80,
               ipady=20)
        btn_upload=Button(file_upload_frame,text="Analyze",command=self.textNew,width=15,font=("times new roman",14,"bold"),bg="lightpink",fg="white").grid(row=2,column=1,padx=20)
        self.root.txtlbl2=Entry(file_upload_frame,width=20,bd=10,textvariable=self.resultp,bg="green",relief=GROOVE,font=("",12),fg="white").grid(row=3,column=1,padx=50,pady=20)
        self.root.txtlbl3=Entry(file_upload_frame,width=20,bd=10,textvariable=self.resultn,bg="red",relief=GROOVE,font=("",12),fg="white").grid(row=4,column=1,padx=50,pady=10)
        txtconfident=Entry(file_upload_frame,width=60,bd=5,textvariable=self.confident,relief=GROOVE,font=("",12)).grid(row=5,column=1,padx=20,pady=20)
        
        btn=Button(root,text="next page",width=15,command=self.next,font=("times new roman",12,"bold"),bg="palevioletred",fg="white")
        btn.place(rely=1.0, relx=1.0, x=0, y=0,anchor=SE)
        windowWidth = root.winfo_reqwidth()
        windowHeight = root.winfo_reqheight()
        print("Width",windowWidth,"Height",windowHeight)
 
# Gets both half the screen width/height and window width/height
        positionRight = int(root.winfo_screenwidth()/4 - windowWidth/4)
        positionDown = int(root.winfo_screenheight()/10 - windowHeight/10)
 
# Positions the window in the center of the page.
        root.geometry("+{}+{}".format(positionRight, positionDown))
        
        '''
        self.labelFrame=ttk.LabelFrame(self.root,text="open a file")
        self.labelFrame.grid(column=0,row=1,padx=20,pady=20)
        btn_upload=Button(self.labelFrame,text="Select File",width=13,command=self.Upload,font=("times new roman",12,"bold"),bg="black",fg="white").grid(row=2,column=1,padx=50,pady=10)
    '''
     
    def Upload(self):
        if self.ufile.get()=="" :
            #messagebox.showerror("error", "please select your file!")
            messagebox.showinfo("Sentiment Analyzer for Mobile Brands","please select your file in .csv file!") 
            self.filename=filedialog.askopenfilename(initialdir="/",title="select a file",filetype=(("csv files (*.csv)","*.csv"),("All files","*.*")))
               
            self.ufile.set(self.filename)
        
            review_df= pd.read_csv(self.filename)
            print(review_df.shape)
            print(review_df.head(2))
            print(self.filename)
            
       
    def analyze(self):
        review_df= pd.read_csv(self.filename)
        
        
        messagebox.showinfo("Sentiment Analysis for Mobile Brands","Please wait for a while, your data is being analyzed!") 
             
        def new_data(x):
            max_len=500
            raw_text=x
            seq= tokenizer.texts_to_sequences([raw_text])
            text_padded = pad_sequences(seq, maxlen=max_len)
            predict_sentences= model.predict_on_batch(text_padded)
            r=label_names[np.argmax(predict_sentences)]
            return r

        review_df['sentiment_label']= review_df['Cust_Reviews'].apply(lambda x:new_data(x))
        labeled=pd.DataFrame(review_df,columns=['Cust_Reviews','Brands','sentiment_label']) 
        messagebox.showinfo("Sentiment Analysis Application","your data has been analyzed!") 
        
        res = messagebox.askyesnocancel('Sentiment Analysis Application','Do you want to save the file?')
        
        if res==True:
            dir = filedialog.askdirectory()
            labeled.to_csv(os.path.join(dir,r'new_data.csv'))
            messagebox.showinfo("information","your data has been saved!") 
            self.savefile.set(os.path.join(dir,r'new_data.csv'))
        print(self.filename)
        
    def bar(self):
        import pandas as pd
         
        #self.filename=filedialog.askopenfilename(initialdir="/",title="select a file",filetype=(("csv files (*.csv)","*.csv"),("All files","*.*")))
        s=self.savefile.get()
        
        df= pd.read_csv(s,encoding='ISO-8859-1')
        
        import seaborn as sb
        import numpy as np
        import matplotlib.pyplot as plt
        
        sb.countplot(x='sentiment_label',data=df)
        plt.show()
        
    def positive(self):  
         
        import matplotlib.pyplot as plt
        s=self.savefile.get()
        
        df= pd.read_csv(s,encoding='ISO-8859-1')
         
        pos = []
        neg = []

        for l in df.sentiment_label:
            if l == 'Negative':
                pos.append(0)
                neg.append(1)
        
            elif l =='Positive':
                pos.append(1)
                neg.append(0)
        
       
        df['Positive']= pos
        df['Negative']= neg
            
        #sb.countplot(x='Positive',data=df)
#plt.bar(x=np.arange(1,2),height=review_df['Polarity'])
        x= df[['Brands', 'Positive']]
        y= x.set_index('Brands')
        z=y.groupby('Brands').mean()
        z.plot.bar(stacked=True)
        plt.show()
        
    def negative(self):
        
        import matplotlib.pyplot as plt
        s=self.savefile.get()
        
        df= pd.read_csv(s,encoding='ISO-8859-1')
        #sb.countplot(x='Negative',data=df)
#plt.bar(x=np.arange(1,2),height=review_df['Polarity'])
        
        pos = []
        neg = []

        for l in df.sentiment_label:
            if l == 'Negative':
                pos.append(0)
                neg.append(1)
        
            elif l =='Positive':
                pos.append(1)
                neg.append(0)
        
       
        df['Positive']= pos
        df['Negative']= neg
        x= df[['Brands', 'Negative']]
        y= x.set_index('Brands')
        z=y.groupby('Brands').mean()

        z.plot.bar(stacked=True)

        plt.show()
        
    def wordcloudpositive(self):
        
        import matplotlib.pyplot as plt
        s=self.savefile.get()
        
        df= pd.read_csv(s)
        
        from wordcloud import WordCloud
        
        pos = df[df.sentiment_label=='Positive']
        pos_r = []
        for t in pos.Cust_Reviews:
            pos_r.append(t)
        pos_r = pd.Series(pos_r).str.cat(sep=' ')
        df['positive']=pos_r
        
           
        wordc = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(pos_r) 
        
        fig = plt.figure(figsize=(12,10))
        fig.canvas.set_window_title('Frequency word of Positive reviews')
        plt.imshow(wordc, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        
    def wordcloudnegative(self):
        
        import matplotlib.pyplot as plt
        s=self.savefile.get()
        
        df= pd.read_csv(s)
        
        from wordcloud import WordCloud
        neg = df[df.sentiment_label=='Negative']
        neg_r = []
        for t in neg.Cust_Reviews:
            neg_r.append(t)
        neg_r = pd.Series(neg_r).str.cat(sep=' ')
        df['negative']=neg_r
            
        wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_r)
        fig = plt.figure(figsize=(12,10))
        fig.canvas.set_window_title('Frequency word of Negative reviews')
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        
        #messagebox.showinfo("information","focus on word that have the largest frequency to monitor your product!") 
        
                          
root=Tk()
obj=sentiment_analysis(root) 
root.mainloop()
