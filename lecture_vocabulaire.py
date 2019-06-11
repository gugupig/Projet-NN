#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
from collections import defaultdict

path = '/Users/bingzhi/Desktop/LI/Projet-NN-master/fr-ud-v1.3/fr-ud-dev.conllu'
path_train = '/Users/bingzhi/Desktop/LI/Projet-NN-master/fr-ud-v1.3/fr-ud-train.conllu'

"""
2-3 des ne compt pas comme mot à taggué

"""
class Indices:

	""" class to handle the correspondances from indices to words 
	and from words to indices correspondances """

	def __init__(self):
		self.id2w = []
		self.w2id = {}

	def get_nb_words(self):
		return len(self.id2w)
	
	def id_from_w(self, w):
		""" 
		returns index of word w if w is not already known,
		otherwise, creates a new index for w and returns it 
		"""
		if w in self.w2id:
			return self.w2id[w]
		
		self.w2id[w] = len(self.id2w)
		self.id2w.append(w)
		
		return self.w2id[w]	


def lecture_corpus(path_train):
	"""
	read the train corpus and replace all the hapax with tag 'NUM' with special word 'UNK'
	return a dictionary with 
	-key: (word,tag)
	-values: occurrences
		

	"""
	dico_corpus = defaultdict(int)
	with open(path_train) as file:
		for line in file:
			if line!='\n' and (not line.startswith('#')):
				line = line.split('\t')[:4]
				idx = line[0]
				word = line[1]
				tag = line[3]
				# the compound like "des" with index like "2-3" are taged in his profond form:
				# 2 de ADP, 3 les DET, so ignore word with idx >=3
				if len(idx) < 3 :
					dico_corpus[(word,tag)]+=1
	
	# replace all the word with occurrence = 1 and tag = NUM with a special word'UNK' 
	# and update idx word correspondances
	for key in list(dico_corpus.keys()):
		indices.id_from_w('UNK')
		if dico_corpus[key]==1 and key[1]=='NUM':
			dico_corpus['UNK'] +=1
			del dico_corpus[key]
		else:
			indices.id_from_w(key[0])

	return dico_corpus


indices = Indices()
dico_corpus =lecture_corpus(path_train)
#print(dico_corpus)
#print(indices.get_nb_words())
voca_size = indices.get_nb_words()
#print(indices.id2w[:10])

def extract_X_Y(path_corpus,indices):
	""""read a corpus conll and return a tuple of (X,Y)
	X is examples: a list of sentences, every sentence is a list of words' id 
	Y is gold tags: a list of tags for correspondant sentence in X
	"""
	examples = []
	gold_tags = []
	with open(path_corpus) as file:
		sentence = []
		sentence_tags = []
		for line in file:			
			if line!='\n' and (not line.startswith('#')):
				line = line.split('\t')[:4]
				#print(line)
				idx = line[0]
				word = line[1]
				tag = line[3]
				if len(idx) <3:
					if word in indices.w2id:
						id = indices.w2id[word]
						sentence.append(id)
					else:
						sentence.append(indices.w2id['UNK'])
					sentence_tags.append(tag)
					#print(sentence)
			elif line == '\n':
				examples.append(sentence)
				gold_tags.append(sentence_tags)
				#print(examples)
				#print(gold_tags)
				sentence = []
				sentence_tags = []

	return examples,gold_tags

X,Y=extract_X_Y(path,indices)
print(X[:2])
print(Y[:2])

def word_vectors(sentence,indices,voca_size):
	"""
	arguments: 
	-sentence is a list of words' id 

	return the word vectors for each word in sentence: 
	-concatenation of 5 one hot vecteurs: word i-2, word i-1, word i, word i+1, word i+z  and 
	-one boolean capitalisation feature(1 maj, 0 min)

	  
	"""
	l = len(sentence)
	phrase = start+ sentence + end
	init = np.zeros(5,voca_size)
	for idx in range():

	pass  


	













				

