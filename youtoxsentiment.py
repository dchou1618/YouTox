#!/usr/bin/env python3

# NLP
import nltk
import pandas as pd
import PIL.Image
from wordcloud import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class YouToxSentiment:
    def __init__(self,text):
        self.text = text
        self.ytSenti = SentimentIntensityAnalyzer()
    # naive sentiment analysis with vader
    def sentiment(self,comment):
        senti = SentimentIntensityAnalyzer()
        positivity = senti.polarity_scores(comment)
        polarities = {positivity["neg"]:"Negative",
                      positivity["neu"]:"Neutral",
                      positivity["pos"]:"Positive"}
        mostProbable = max(positivity["neg"],
                           max(positivity["neu"],positivity["pos"]))
        conclusion = polarities[mostProbable]
        return conclusion

    def getPolarityWords(self,text):
        polarities = {}
        for comment in text:
            words = comment.split()
            for word in words:
                wordSenti = self.sentiment(word)
                polarities[wordSenti] = polarities.get(wordSenti,[]) + [word]
        self.polarities = polarities
        return polarities

    def plotsYT(self,text,mask,color = 'white'):
        def pngToGIF(path):
            img = PIL.Image.open(path)
            img.save('pic.gif')
        words = " ".join(text)
        self.getPolarityWords(text)
        try:
            positives = self.polarities["Positive"]
            negatives = self.polarities["Negative"]
        except:
            return "Error"
        wordcloud1 = WordCloud(max_words = 100,
                              stopwords=STOPWORDS,
                              background_color=color,
                              width=100,height=100,
                              collocations = False,
                              mask=mask).generate(words)
        wordcloud2 = WordCloud(max_words = 100,
                              stopwords=STOPWORDS,
                              background_color=color,
                              width=100,height=100,
                              collocations = False,
                              mask=mask).generate(" ".join(positives))
        wordcloud3 = WordCloud(max_words = 100,
                              stopwords=STOPWORDS,
                              background_color=color,
                              width=100,height=100,
                              collocations = False,
                              mask=mask).generate(" ".join(negatives))
        # matches text to colors from original mask
        cloudColors = ImageColorGenerator(mask)
        # constructing subplots
        fig = plt.figure(figsize=(8,8))
        rows,columns = 2,2
        fig.add_subplot(rows,columns,1)
        # displaying plot of basic sentiment distribution
        lstSenti = [self.sentiment(comment) for comment in text]
        sentiments = pd.DataFrame({"Sentiment":lstSenti})
        sns.countplot(x="Sentiment",data=sentiments)
        plt.title("Youtube Video Sentiment")

        # displaying word cloud of overall headlines
        fig.add_subplot(rows,columns,2)
        plt.imshow(wordcloud1.recolor(color_func=cloudColors),
                   interpolation='bilinear')
        plt.title("Youtube Video Comments")

        # display most common positive words
        fig.add_subplot(rows,columns,3)
        plt.imshow(wordcloud2.recolor(color_func=cloudColors),
                   interpolation='bilinear')
        plt.xlabel("Youtube Video Positive")

        # display most common negative words
        fig.add_subplot(rows,columns,4)
        plt.imshow(wordcloud3.recolor(color_func=cloudColors),
                   interpolation='bilinear')
        plt.xlabel("Youtube Video Negative")
        plt.savefig("pic.png")
        plt.clf()
        pngToGIF("pic.png")
    def completedGroups(self,currGroups):
        return None not in currGroups
    def validGroups(self,currGroups,i,sentence,L):
        pass
    def groupWrapper(self,currGroups,allItemGroups,L):
        if self.completedGroups(currGroups):
            return currGroups
        for i in range(len(currGroups)):
            for sentence in L:
                if validGroups(currGroups,i,sentence,L):
                    currGroups[i].append(sentence)
                    tempGroups = groupWrapper(self,currGroups,allItemGroups,L)
                    if tempGroups != None:
                        return tempGroups
                    currGroups[i].pop()

    # generates different groups of related-toxicity labels
    def generateGroups(self,L,n):
        currGroups = [None]*6 # number of toxicity labels
        allItemGroups = []
        return self.groupWrapper(currGroups,allItemGroups,L,n)

def analyzeSentiment(words):
    yt = YouToxSentiment(words)
    yt.plotsYT()
