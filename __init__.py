#!/usr/bin/env python3

import tkinter as tk
import numpy as np

import PIL.Image
import PIL.ImageTk
import PIL.ImageSequence
import copy
import os
import threading
import cv2
from tkinter import *
from youToxDataBase import *

class Bar:
    def __init__(self,x0,y0,x1,y1):
        self.x0 = x0; self.y0 = y0
        self.x1 = x1; self.y1 = y1
    def draw(self,canvas,fill):
        canvas.create_rectangle(self.x0,self.y0,self.x1,self.y1,fill=fill,\
                                width=10)
    def inBar(self,x,y):
        return self.x0<=x<=self.x1 and self.y0<=y<=self.y1

class SearchBar(Bar):
    def __init__(self,x0,y0,x1,y1):
        super().__init__(x0,y0,x1,y1)

    def draw(self,canvas,fill,width,height,typing,query):
        super().draw(canvas,fill)
        size = self.y1-self.y0
        searchX0 = 3*(self.x0+self.x1)//4
        self.searchX0 = searchX0
        canvas.create_rectangle(searchX0,self.y0,
                                self.x1,self.y1,fill="gold",outline=None)
        canvas.create_text((searchX0+self.x1)//2,(self.y0+self.y1)//2,
                            text="Search",fill="darkBlue",
                            font = "Merriweather "+str((self.x1-self.x0)//20))
        margin = 5
        if typing:
            canvas.create_text((self.x0+self.searchX0)//2,
                                (self.y0+self.y1)//2,text=str(query),
                                anchor="center",
                                font="Helvetica "+str(int(size*0.25))+" bold")

    def inSearchBar(self,x,y):
        return self.x0<=x<=self.searchX0 and self.y0<=y<=self.y1
    def inSearchButton(self,x,y):
        return self.searchX0<=x<=self.x1 and self.y0<=y<=self.y1

class NavigationBar(Bar):
    items = ["Home","About","Youtube"]
    def __init__(self,x0,y0,x1,y1):
        super().__init__(x0,y0,x1,y1)
    def draw(self,canvas,fill):
        super().draw(canvas,fill)
        self.width = (self.x0+self.x1)*0.2
        self.height = self.y1-self.y0
        for i in range(4):
            centerY = self.height//2
            centerX = (i*self.width+(i+1)*self.width)/2
            if i < 3:
                canvas.create_rectangle(i*self.width,0,
                                        (i+1)*self.width,
                                        self.height,fill=fill)
                canvas.create_text(centerX,centerY,text=NavigationBar.items[i])
            else:
                centerX = (i*self.width+(i+2)*self.width)/2
                canvas.create_rectangle(i*self.width,0,
                                        (i+2)*self.width,
                                        self.height,fill="lightgreen")
                canvas.create_text(centerX,centerY,text="YouTox",fill="black",
                                   font = "Helvetica "+str(int(self.height//2))+" bold")
    def inWhichItem(self,x,y):
        for i in range(3):
            if i*self.width<=x<=(i+1)*self.width and 0<=y<=self.height:
                return NavigationBar.items[i]

class ToxPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def dist(self,other):
        try:
            assert(isinstance(other,ToxPoint))
            distance = ((self.x-other.x)**2+(self.y-other.y)**2)**0.5
            return distance
        except:
            return

def depictPredictSentiment(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # https://www.nltk.org/api/nltk.sentiment.html
    myEchoSentiment = SentimentIntensityAnalyzer()
    positivity = myEchoSentiment.polarity_scores(text)
    polarVals = {positivity["neg"]:"Negative",
                  positivity["neu"]:"Neutral",
                  positivity["pos"]:"Positive"}
    highestMagnitude = max(positivity["neg"],
                       max(positivity["neu"],positivity["pos"]))
    return highestMagnitude

def init(d):
    import pandas as pd
    import seaborn as sns
    try:os.remove("pic.gif");os.remove("pic.png")
    except:pass
    try:os.remove("toxicity.gif");os.remove("toxicity.png")
    except:pass
    try:os.remove("gNews.gif");os.remove("gNews.png")
    except:pass
    d.mode = "HomePage"; d.modes = ["HomePage","AboutPage","YoutubePage",\
                                    "ToxPage","YTPage","GooglePage","LoadingPage"]
    d.query = ""; d.maxLength = 50
    d.typingHome = False
    d.typingYoutube = False
    d.description = "YouTox is an application that labels and\n"+\
                    "summarizes toxicity and polarity of comments.\n\nEnter a "+\
                    "comment to analyze toxicity and sentiment"
    d.prompt = "Enter a URL to analyze comment toxicity"
    d.about = "YouTox is tkinter application that employs\nlogistic regression and " + \
              "sentiment analysis\nto determine polarity of comments"
    d.showCursorYoutube = False
    d.showCursorHome = False
    d.second = 0
    d.curr = "StartPage"
    d.navigation = NavigationBar(0,0,d.width,50)
    d.loadedAmount = 0
    d.loading = False
    d.nextMode = ""
    d.dataStorage = YouToxDB()

    d.comments = pd.read_csv("train.csv")
    i = 0
    for index,row in d.comments.iterrows():
        i += 1
        if i > 100:
            break
        for val in list(d.dataStorage.sentencesTox.keys()):
            if val == "maxTox":
                d.dataStorage.sentencesTox["maxTox"].append(\
                depictPredictSentiment(row["comment_text"]))
            elif val == "length":
                d.dataStorage.sentencesTox[val].append(len(row["comment_text"]))
            elif val == "text":
                d.dataStorage.sentencesTox[val].append(row["comment_text"])
        toxVal = None; addedTox = False
        for key in list(row.keys())[::-1]:
            if row[key] == 1 and not addedTox:
                toxVal = key
                addedTox = True
        d.dataStorage.sentencesTox["tox"].append(toxVal)
    d.winnieGif = [PIL.ImageTk.PhotoImage(img) for img in \
                  PIL.ImageSequence.Iterator(PIL.Image.open("winnie.gif"))]
    d.winnieIndex = 0

    if "YouToxLogistictoxic.sav" in os.listdir("."): d.trainModel = False
    else: d.trainModel = True

def drawStartPage(c,d):
    d.navigation.draw(c,"lightblue")
    margin = 10
    s = SearchBar(margin,d.height*0.5,d.width-margin,d.height*0.6)
    d.startSearch = s
    s.draw(c,None,d.width,d.height,d.typingHome,d.query)
    c.create_text(d.width//2,d.height//3,text=d.description,
                  font="Helvetica "+str(int(d.height*0.035))+" bold")
    if d.showCursorHome:
        centerX = d.width*0.385
        c.create_line(centerX+d.height*0.035/3.5*len(d.query)*0.6,d.height*0.5,
                      centerX+d.height*0.035/3.5*len(d.query)*0.6,d.height*0.6)
    drawNewsButton(c,d)

def drawAboutPage(c,d):
    pic = PhotoImage(file="dataPic.gif")
    img = Label(image=pic)
    img.image = pic
    c.create_image(d.width/2,d.height*0.60,image=pic)
    d.navigation.draw(c,"lightblue")
    margin = 10
    c.create_text(d.width*0.45,d.height//7,text=d.about,
                  font="Merriweather "+str(int(d.height*0.035)) + " bold")

def drawYoutubePage(c,d):
    d.navigation.draw(c,"lightblue")
    margin = 10
    s = SearchBar(margin,d.height*0.5,d.width-margin,d.height*0.6)
    d.youtubeSearch = s
    s.draw(c,None,d.width,d.height,d.typingYoutube,d.query)
    c.create_text(d.width//2,d.height//3,text=d.prompt,
                  font="Merriweather "+str(int(d.height*0.035))+" bold")
    if d.showCursorYoutube:
        centerX = d.width*0.385
        c.create_line(centerX+d.height*0.035/3.5*len(d.query)*0.6,d.height*0.5,
                      centerX+d.height*0.035/3.5*len(d.query)*0.6,d.height*0.6)

def drawNewsButton(c,d):
    button = Bar(d.width//3,d.height*0.65,2*d.width//3,d.height*0.75)
    button.draw(c,'lightblue')
    centerX,centerY = d.width/2,(d.height*0.65+d.height*0.75)/2
    c.create_text(centerX,centerY,text="Today's News",
                  font="Merriweather "+str(int(d.height*0.035)))
    d.newsButton = button

def drawytPage(c,d):
    c.delete("all")
    d.navigation.draw(c,"lightblue")
    pic = PhotoImage(file="pic.gif")
    img = Label(image=pic)
    img.image = pic
    c.create_image(d.width/2,d.height*0.45,image=pic)

def drawToxPage(c,d):
    c.delete("all")
    d.navigation.draw(c,"lightblue")
    pic = PhotoImage(file="toxicity.gif")
    pic = pic.subsample(2,2)
    img = Label(image=pic)
    img.image = pic
    c.create_image(d.width*0.35,d.height*0.30,image=pic)
    drawCorrelation(c,d)

def drawGooglePage(c,d):
    c.delete("all")
    d.navigation.draw(c,"lightblue")
    pic = PhotoImage(file="gNews.gif")
    img = Label(image=pic)
    img.image = pic
    c.create_image(d.width*0.10,d.height*0.50,image=pic)

def drawLoadingPage(c,d):
    c.delete("all")
    winnie = d.winnieGif[d.winnieIndex]
    img = Label(image=winnie)
    img.image = winnie
    c.create_image(d.width/2,d.height*0.45,image=winnie)
    d.navigation.draw(c,"lightblue")
    c.create_text(d.width/2,d.height*0.70,text="Loading",font="Merriweather "+\
                  str(int(d.width*0.05))+" bold")
    c.create_rectangle(0.05*d.width,0.75*d.height,d.width*0.05+(d.loadedAmount),\
                       0.85*d.height,fill="gold")
    c.create_rectangle(0.05*d.width,0.75*d.height,d.width*0.95,0.85*d.height,\
                       fill=None,width=10)

def getEuclideanDist(point1,point2):
    return point1.dist(point2)

def drawCorrelation(c,d):
    import seaborn as sns
    import matplotlib.pyplot as plt1
    import pandas as pd
    # print(d.dataStorage.sentencesTox["tox"])
    # print(d.dataStorage.sentencesTox["maxTox"],d.dataStorage.sentencesTox["length"])
    sns.scatterplot(x="length",y="maxTox",hue="tox",\
                          data=d.dataStorage.sentencesTox)
    plt1.title("Toxic Magnitude vs. Length of Comments")
    plt1.xlabel("Length of Text")
    plt1.ylabel("Magnitude of Toxicity")

    plt1.savefig("correlation.png")
    img = PIL.Image.open("correlation.png")
    img.save("correlation.gif")

    corr = PhotoImage(file="correlation.gif")
    corr = corr.subsample(2,2)
    img1 = Label(image=corr)
    img1.image = corr
    c.create_image(d.width*0.35,d.height*0.70,image=corr)

def timerLoadingPage(d):
    if d.loading:
        d.loadedAmount += d.width*0.0025
        if d.loadedAmount >= 0.90*d.width:
            d.loading = False
            d.mode = d.nextMode
            d.loadedAmount = 0

def timerFired(d):
    d.second += 1
    d.winnieIndex += 1
    d.winnieIndex %= len(d.winnieGif)

    if d.second%5 == 0:
        if d.mode == "HomePage":
            if d.typingHome:
                d.showCursorHome = not d.showCursorHome
        elif d.mode == "YoutubePage":
            if d.typingYoutube:
                d.showCursorYoutube = not d.showCursorYoutube
    if d.mode == "LoadingPage":
        timerLoadingPage(d)

def editQuery(e,d,i,typing):
    import youToxLogistic, youtubeComments, youtoxsentiment
    if typing and len(d.query) < d.maxLength:
        if e.keysym == "BackSpace":
            d.query = d.query[:-1]
        elif e.keysym == "Return":
            if i == 1:
                if "www.youtube" in d.query:
                    d.mode = "YTPage"
                    comments = youtubeComments.runCommentScrape(d.query)
                    yt = youtoxsentiment.YouToxSentiment(d.query)
                    yt.plotsYT(comments,np.array(PIL.Image.open("ytLogo.png")))
                else:
                    d.mode = "HomePage"
            else:
                d.mode = "ToxPage"
                youToxLogistic.run(d.query)
        else:
            d.query += e.char
    else:
        if e.keysym == "BackSpace":
            d.query = d.query[:-1]

def keyPressed(e,d):
    if d.mode == "HomePage":
        editQuery(e,d,0,d.typingHome)
    if d.mode == "YoutubePage":
        editQuery(e,d,1,d.typingYoutube)

def changePage(e,d):
    for i in range(len(d.modes)):
        item = d.navigation.inWhichItem(e.x,e.y)
        if item != None and item in d.modes[i]:
            d.mode = d.modes[i]; d.q = d.query
            d.query = ""
            d.typingHome = False; d.typingYoutube = False
            d.showCursorYoutube = False; d.showCursorHome = False

def mousePressed(e,d):
    import youToxLogistic, googleNewsScraper, youtubeComments
    import youtoxsentiment
    if d.mode == "HomePage":
        if d.startSearch.inSearchBar(e.x,e.y):
            d.typingHome = True
        elif d.startSearch.inSearchButton(e.x,e.y):
            d.nextMode = "ToxPage"
            result = []
            logisticModel = threading.Thread(target = youToxLogistic.run,\
                                             args = (d.query,d.trainModel,result))
            logisticModel.start()
            d.mode = "LoadingPage"
            d.trainModel = False
            d.loading = True
        elif d.newsButton.inBar(e.x,e.y):
            d.nextMode = "GooglePage"
            googleNews = threading.Thread(target = googleNewsScraper.scrapeGoogleNews)
            googleNews.start()
            d.mode = "LoadingPage"
            d.loading = True
        changePage(e,d)
    elif d.mode == "YoutubePage":
        if d.youtubeSearch.inSearchBar(e.x,e.y):
            d.typingYoutube = True
        if d.youtubeSearch.inSearchButton(e.x,e.y):
            if "www.youtube" in d.query:
                d.nextMode = "YTPage"
                comments = youtubeComments.runCommentScrape(d.query)
                yt = youtoxsentiment.YouToxSentiment(d.query)
                yt.plotsYT(comments,np.array(PIL.Image.open("ytLogo.png")))
                d.mode = "LoadingPage"
                d.loading = True
            else:
                d.mode = "HomePage"
        changePage(e,d)
    elif d.mode == "AboutPage":
        changePage(e,d)
    elif d.mode == "ToxPage":
        changePage(e,d)
    elif d.mode == "YTPage":
        changePage(e,d)
    elif d.mode == "GooglePage":
        changePage(e,d)

def redrawAll(c,d):
    if d.mode == "HomePage":
        drawStartPage(c,d)
    elif d.mode == "AboutPage":
        drawAboutPage(c,d)
    elif d.mode == "YoutubePage":
        drawYoutubePage(c,d)
    elif d.mode == "ToxPage":
        drawToxPage(c,d)
    elif d.mode == "YTPage":
        drawytPage(c,d)
    elif d.mode == "GooglePage":
        drawGooglePage(c,d)
    elif d.mode == "LoadingPage":
        drawLoadingPage(c,d)
    image = PhotoImage(file="logo.gif")
    image = image.subsample(11,11)
    img = Label(image=image)
    img.image = image
    c.create_image(d.width*620/700,d.height*2.5/700,anchor=NW,image=image)
    c.create_image(d.width*460/700,d.height*2.5/700,anchor=NW,image=image)

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 50 # milliseconds
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    root.mainloop()

if __name__ == "__main__":
    run(1000, 1000)
