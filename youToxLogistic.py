# basic dependencies
import string
import re
import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning) # no future warnings
# regressional models
from sklearn.linear_model import LogisticRegression
# NLP vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# visualizations
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import os
import seaborn as sns
import PIL.Image
# classification of multiple labels
# assuming no correlation between labels of toxicity in comment
# implemented using csv file data from
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
class YouToxLogistic:
    youtoxlog = LogisticRegression(C=12.0)
    toxicities = ["insult","toxic","obscene","severe_toxic","threat","identity_hate"]
    def __init__(self,**kwargs):
        for key in kwargs:
            val = kwargs[key]
            if key == "data":
                exec("self.{}=pd.read_csv('{}')".format(key,val))
            else:
                exec("self.{}=\"{}\"".format(key,val))
    def cleanData(self):
        def cleanText(text):
            text = text.lower()
            text = re.sub(r"\'s"," ",text)
            text = re.sub(r"\'ll","will",text)
            text = re.sub(r"\'d","would",text)
            text = text.encode('ascii', 'ignore').decode('ascii')
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\W', ' ', text)
            text = text.strip()
            return text
        try:
            self.data["comment_text"] = self.data["comment_text"].map(lambda txt: cleanText(txt))
        except Exception as e:
            return "Error with the data!"+str(e)
    def train(self):
        def predictTox(val):
            return YouToxLogistic.youtoxlog.predict_proba(val)[:,1]
        def pngToGIF(path):
            img = PIL.Image.open(path)
            img.save('toxicity.gif')
        self.cleanData()
        # data vectorization
        limit = 5000
        txtvect = TfidfVectorizer(max_features = limit, stop_words="english")
        # transformed x training data
        xTrain = txtvect.fit_transform(self.data.comment_text)
        # vectorize val passed in for prediction
        statement = self.val
        xVal = pd.Series(self.val)
        self.val = txtvect.transform(xVal)
        # ^^ transforms the dtype array only
        toxData = dict()
        for tox in YouToxLogistic.toxicities:
            yTrain = self.data[tox]
            # fitting the data to logistic regression
            YouToxLogistic.youtoxlog.fit(xTrain,yTrain)
            # predicted probabilities of matching labels
            estimatedYProb = predictTox(self.val)
            toxData[tox] = estimatedYProb
        self.toxPredict = toxData
        yValues = [val[0]*100 for val in self.toxPredict.values()]
        sns.barplot(x=YouToxLogistic.toxicities,y=yValues)
        plt.title("Toxicity of Corpus")
        plt.xlabel("Toxicity Scale")
        plt.ylabel("% Probability of Label")
        plt.savefig('toxicity.png')
        # turns png image into gif
        pngToGIF("toxicity.png")

def run(val):
    yt = YouToxLogistic(data="./train.csv", val = val)
    yt.train()
