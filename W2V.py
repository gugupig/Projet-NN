# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 00:11:28 2019

@author: gugup
"""

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import pickle
pre_train = KeyedVectors.load_word2vec_format('frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin', binary=True)

with open('st.pkl','rb') as file:
    snss = pickle.load(file)
wiki_50 = KeyedVectors.load_word2vec_format('vecs50-linear-frwiki', binary=False)

model = Word2Vec(snss,size = 10,min_count = 1,window = 5,workers = 8)

model.wv['facile']

model_2 = Word2Vec(size = 200,min_count = 1,window =5,workers = 8)
model_2.build_vocab(snss)
total_examples = model_2.corpus_count
model_2.build_vocab([list(pre_train.vocab.keys())],update = True)
model_2.intersect_word2vec_format('frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin',binary = True,lockf =1.0)
model_2.train(snss,total_examples = total_examples,epochs = model_2.iter)
model_2.wv['facile'].shape

model_3 = Word2Vec(size = 50,min_count = 1,window =5,workers = 8)
model_3.build_vocab(snss)
total_examples = model_3.corpus_count
model_3.build_vocab([list(wiki_50.vocab.keys())],update = True)
model_3.intersect_word2vec_format('vecs50-linear-frwiki',binary = False,lockf =1.0)
model_3.train(snss,total_examples = total_examples,epochs = model_3.iter)
model_3.wv['facile'].shape

model.wv.save_word2vec_format('model_10.bin', binary=True)
model_2.wv.save_word2vec_format('model_200.bin', binary=True)
model_3.wv.save_word2vec_format('model_50.bin', binary=True)