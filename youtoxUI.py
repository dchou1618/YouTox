#!/usr/bin/env python3

import tkinter as tk
import numpy as np
import PIL.Image
import copy
import os
from tkinter import *

class Bar:
    def __init__(self,x0,y0,x1,y1):
        self.x0 = x0; self.y0 = y0
        self.x1 = x1; self.y1 = y1
    def draw(self,canvas,fill):
        canvas.create_rectangle(self.x0,self.y0,self.x1,self.y1,fill=fill)
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

def init(d):
    d.mode = "HomePage"; d.modes = ["HomePage","AboutPage","YoutubePage","DataPage"]
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

def drawStartPage(c,d):
    n = NavigationBar(0,0,d.width,50)
    d.navigationBar = n
    n.draw(c,"lightblue")
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

    n = NavigationBar(0,0,d.width,50)
    d.navigationBar = n
    n.draw(c,"lightblue")
    margin = 10
    c.create_text(d.width*0.45,d.height//7,text=d.about,
                  font="Merriweather "+str(int(d.height*0.035)) + " bold")

def drawYoutubePage(c,d):
    n = NavigationBar(0,0,d.width,50)
    n.draw(c,"lightblue")
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

def drawDataPage(c,d):
    n = NavigationBar(0,0,d.width,50)
    n.draw(c,"lightblue")
    margin = 10
    pic = PhotoImage(file="pic.gif")
    img = Label(image=pic)
    img.image = pic
    c.create_image(d.width/2,d.height*0.45,image=pic)

def timerFired(d):
    d.second += 1
    if d.second%5 == 0:
        if d.mode == "HomePage":
            if d.typingHome:
                d.showCursorHome = not d.showCursorHome
        elif d.mode == "YoutubePage":
            if d.typingYoutube:
                d.showCursorYoutube = not d.showCursorYoutube

def editQuery(e,d,i,typing):
    import youToxLogistic, youtubeComments, youtoxsentiment
    if typing and len(d.query) < d.maxLength:
        if e.keysym == "BackSpace":
            d.query = d.query[:-1]
        elif e.keysym == "Return":
            d.mode = "DataPage"
            if i == 1:
                if "www.youtube" in d.query:
                    comments = youtubeComments.runCommentScrape(d.query)
                    yt = youtoxsentiment.YouToxSentiment(d.query)
                    yt.plotsYT(comments,np.array(PIL.Image.open("ytLogo.png")))
                else:
                    d.mode = "HomePage"
            else:
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
        item = d.navigationBar.inWhichItem(e.x,e.y)
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
        if d.startSearch.inSearchButton(e.x,e.y):
            d.mode = "DataPage"
            youToxLogistic.run(d.query)
        if d.newsButton.inBar(e.x,e.y):
            d.mode = "DataPage"
            googleNewsScraper.scrapeGoogleNews()
        changePage(e,d)
    elif d.mode == "YoutubePage":
        if d.youtubeSearch.inSearchBar(e.x,e.y):
            d.typingYoutube = True
        if d.youtubeSearch.inSearchButton(e.x,e.y):
            d.mode = "DataPage"
            if "www.youtube" in d.query:
                comments = youtubeComments.runCommentScrape(d.query)
                yt = youtoxsentiment.YouToxSentiment(d.query)
                yt.plotsYT(comments,np.array(PIL.Image.open("ytLogo.png")))
            else:
                d.mode = "HomePage"
        changePage(e,d)
    elif d.mode == "AboutPage":
        changePage(e,d)
    elif d.mode == "DataPage":
        changePage(e,d)

def redrawAll(c,d):
    if d.mode == "HomePage":
        drawStartPage(c,d)
    elif d.mode == "AboutPage":
        drawAboutPage(c,d)
    elif d.mode == "YoutubePage":
        drawYoutubePage(c,d)
    elif d.mode == "DataPage":
        drawDataPage(c,d)
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
    data.timerDelay = 100 # milliseconds
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
