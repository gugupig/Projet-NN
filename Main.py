# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:42:39 2019

@author: Kurros
"""
import re
import numpy as np
#import cupy as cp
import pickle
import NN_Class_Alpha_01 as NET
#import cuda_test as NET_G
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import random
import nltk

path_test = 'fr-ud-test.conllu'
path_train = 'fr-ud-train.conllu'

model_10 = KeyedVectors.load_word2vec_format('model_10.bin', binary=True)
model_50 = KeyedVectors.load_word2vec_format('model_50.bin', binary=True)
model_200 = KeyedVectors.load_word2vec_format('model_200.bin', binary=True)







'''
with open(path_train,'r',encoding = 'utf-8') as file:
    for line in file:
        if line.startswith('#') and line[6] == 'e':
                           #i = 0
                           #sentence = nltk.word_tokenize(line.lower().split(':',1)[1]) 
                           #sentences.append(sentence)
                           sentence = []
        if line!='\n' and (not line.startswith('#')):
                           line = line.split('\t')[:4]
                           tag = line[3]
                           cap = line[1].isupper()
                           word = line[1].lower()
                           sentence.append(line[1])
                           if tag not in tags:
                               tags.append(tag)
                           if word not in words:
                               words.append(word)
                           if len(sentence)<5:
                               pass
                           else:
                               if i == 0:
                                   ww = ([word,'N/A','N/A',sentence[1],sentence[2]],tag)
                               elif i == 1:
                                   ww =([word,'N/A',sentence[0],sentence[2],sentence[3]],tag)
                               elif i == len(sentence):
                                   ww = ([word,sentence[-2],sentence[-3],'N/A','N/A'],tag)
                               elif i == len(sentence) -1 :
                                   ww = ([word,sentence[-3],sentence[-4],sentence[-1],'N/A'],tag)
                               else:
                                   ww = ([word,sentence[i-2],sentence[i-1],sentence[i+1],sentence[i+2]],tag)
                               print(i)
                               i= i+1
                               word_windows.append(ww)
                                               
'''                                              
sentences = []
word_windows= []
tags = []
words = []
ww = []
sentence = []
with open(path_train,'r',encoding = 'utf-8') as file:
    for line in file:
        #if line.startswith('#'):
                           
                           #sentence = []
        if line!='\n' and (not line.startswith('#')):
                           line = line.split('\t')[:4]
                           tag = line[3]
                           cap = line[1].isupper()
                           word = line[2].lower()
                           sentence.append((word,tag))
                           if tag not in tags:
                               tags.append(tag)
                           if word not in words:
                               words.append(word)
        if line == '\n':
            sentences.append(sentence)
            sentence = []


#tags.append('UNK')                                                  
random.shuffle(words)
random.shuffle(tags)
random.shuffle(sentences)

                                                        

for s in sentences:
    if len(s) < 5:
        pass
    else:
        for i in range(len(s)):
            if i == 0:
                ww = ([s[0][0],'N/A','N/A',s[1][0],s[2][0]],s[0][1])
            elif i == 1:
                ww = ([s[1][0],'N/A',s[0][0],s[2][0],s[3][0]],s[1][1])
            elif i == len(s)-1:
                ww = ([s[-1][0],s[-3][0],s[-2][0],'N/A','N/A'],s[-1][1])
            elif i == len(s)-2:
                ww = ([s[-2][0],s[-4][0],s[-3][0],s[-1][0],'N/A'],s[-2][1])
            else:
                ww = ([s[i][0],s[i-2][0],s[i-1][0],s[i+1][0],s[i+2][0]],s[i][1])
            word_windows.append(ww)
random.shuffle(word_windows)


examples = []
tag_dim = len(tags)
for ww in word_windows:
    x = []
    tag_id = tags.index(ww[1])
    y = np.zeros(tag_dim)
    y[tag_id] = 1.0
    for word in ww[0]:
        if word not in model_10.wv.vocab and word != 'N/A':
            pass
        elif word == 'N/A':
            x.append(list(np.zeros(10)))
        else:
            x.append(list(model_10.wv[word]))
    if len(x) == 5:
        x = (np.array(np.array(x).reshape(50,))).astype(float)
        examples.append((x,y))
random.shuffle(examples)


sentences_test = []
word_windows_test= []
sentence = []
with open(path_test,'r',encoding = 'utf-8') as file:
    for line in file:
        #if line.startswith('#'):
                           
                           #sentence = []
        if line!='\n' and (not line.startswith('#')):
                           line = line.split('\t')[:4]
                           tag = line[3]
                           cap = line[1].isupper()
                           word = line[2].lower()
                           sentence.append((word,tag))
                           #if tag not in tags:
                               #tags.append(tag)
                           #if word not in words:
                               #words.append(word)
        if line == '\n':
            sentences_test.append(sentence)
            sentence = []
q = ('_', '_')
for s in sentences_test:
    s = [x for x in s if x != q]
    

#tags.append('UNK')                                                  

random.shuffle(sentences_test)

                                                        

for s in sentences_test:
    if len(s) < 5:
        pass
    else:
        for i in range(len(s)):
            if i == 0:
                ww = ([s[0][0],'N/A','N/A',s[1][0],s[2][0]],s[0][1])
            elif i == 1:
                ww = ([s[1][0],'N/A',s[0][0],s[2][0],s[3][0]],s[1][1])
            elif i == len(s)-1:
                ww = ([s[-1][0],s[-3][0],s[-2][0],'N/A','N/A'],s[-1][1])
            elif i == len(s)-2:
                ww = ([s[-2][0],s[-4][0],s[-3][0],s[-1][0],'N/A'],s[-2][1])
            else:
                ww = ([s[i][0],s[i-2][0],s[i-1][0],s[i+1][0],s[i+2][0]],s[i][1])
            word_windows_test.append(ww)
            
random.shuffle(word_windows_test)


examples_test = []
tag_dim_t = len(tags)
for ww in word_windows_test:
    x = []
    tag_id = tags.index(ww[1])
    y = np.zeros(tag_dim)
    y[tag_id] = 1.0
    for word in ww[0]:
        if word not in model_10.wv.vocab and word != 'N/A':
            pass
        elif word == 'N/A':
            x.append(list(np.zeros(10)))
        else:
            x.append(list(model_10.wv[word]))
    if len(x) == 5:
        x = (np.array(np.array(x).reshape(50,))).astype(float)
        examples_test.append((x,y))  
random.shuffle(examples_test)            



result = [] 
new = NET.NN([50],'sigmoid','softmax','cross')
exs_new = examples
new.mini_batch_train(exs_new,len(examples),100,500,0.1)
for test in examples_test:
    predit = new.predit(test[0])
    result.append((predit,test[1]))

res = []
t = 0
for r in result:
    if np.argmax(r[0]) == (np.nonzero(r[1])[0][0]):
        t += 1
        res.append(True)
    else :
        res.append(False)
print (float(t)/float(len(result)))