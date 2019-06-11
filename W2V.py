# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 00:11:28 2019

@author: gugup
"""

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
model = KeyedVectors.load_word2vec_format('vecs50-linear-frwiki', binary=True)
vector = model['facile']
print(vector)