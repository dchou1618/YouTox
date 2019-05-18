#!/usr/bin/env python3

class YouToxDB:
    def __init__(self):
        self.sentencesTox = {val:[] for val in ["length",\
                             "comment_text","maxTox","tox"]}
    def delete(self,key):
        del self.sentencesTox[key]
    def addKey(self,key,sentence):
        self.sentencesTox[key] = self.sentencesTox.get(key,[])+[sentence]
