import numpy as np
import random
import pickle
from gensim.models import KeyedVectors


"""
This module is responsible for building training, test and validation set.
Main Components :
1.get_tags: build a list of tags for indexing
2.make_dense_examples: build a N*D+1 matrix (examples) from corpus 
where N = number of examples,D = number of dimension of the embedding vectors, +1 = tag ids
3.make_sparse_examples: same utility as the last function but return 2 matrix,where x is an N*D matrix of embedding vectors 
and y is an N*T matrix of the one-hot encoded tag  (T = number of  different tags)
"""

path_test = 'fr-ud-test.conllu'
path_train = 'fr-ud-train.conllu'
path_dev = 'fr-ud-dev.conllu'

model_10 = KeyedVectors.load_word2vec_format('model_10.bin', binary=True)


def get_tags(path):
    tags = []
    sentence_tags = []
    buffer = []
    with open(path,'r',encoding = 'utf-8') as file:
        for line in file:
            if line.startswith('#') and line[6] == 'e' :
                               sentence_tags.append(buffer)
                               buffer = []
            if line!='\n' and (not line.startswith('#')):
                               line = line.split('\t')[:4]
                               tag = line[3]
                               if tag != '_':
                                   buffer.append(tag)
                               if tag not in tags and tag!= '_':
                                   tags.append(tag)
    sentence_tags.append(buffer)
    sentence_tags = sentence_tags[1:]
    tags.append('UNK')
    random.shuffle(tags)
    return tags,sentence_tags

def UNK_replace(sentences,sentence_tags):
    for x,y in zip(sentences,sentence_tags):
        for i  in range(len(x)):
            if x[i] == 'UNK': y[i] = 'UNK'



def dump(file_name,obj):
    with open (file_name,'rb') as file:
        pickle.dump(obj,file)
        
def load(var_name,file_name):
    with open (file_name,'wb') as file:
        var_name = pickle.load(file)
        return var_name

    
def word2vec(word,model):
    if word not in model.wv.vocab:
        return model.wv['UNK']
    #
    else:
        return model.wv[word]
    
    
def make_dense_examples(sentences,sentence_tags,tags,model,test = False):
    
    if test == False:
        word_windows = np.zeros((1,31))
        for s,t in zip(sentences,sentence_tags):
            if len(s)<3:
                pass
            else:
                for i in range(len(s)):
                    if i == 0:
                        main_word = word2vec(s[0],model).reshape(1,-1)
                        prev_word = np.zeros((1,10))
                        next_word = word2vec(s[1],model).reshape(1,-1)
                        tag_id = tags.index(t[0])
                        window = np.concatenate((main_word,prev_word,next_word),axis = 1)
                        window = np.append(window,tag_id).reshape(1,-1)
                    if i == len(s) -1 :
                        main_word = word2vec(s[-1],model).reshape(1,-1)
                        prev_word = word2vec(s[-2],model).reshape(1,-1)
                        next_word = np.zeros((1,10))
                        tag_id = tags.index(t[-1])
                        window = np.concatenate((main_word,prev_word,next_word),axis = 1)
                        window = np.append(window,tag_id).reshape(1,-1)
                    else:
                        main_word = word2vec(s[i],model).reshape(1,-1)
                        prev_word = word2vec(s[i-1],model).reshape(1,-1)
                        next_word = word2vec(s[i+1],model).reshape(1,-1)
                        tag_id = tags.index(t[i])
                        window = np.concatenate((main_word,prev_word,next_word),axis = 1)
                        window = np.append(window,tag_id).reshape(1,-1)
                    word_windows = np.concatenate((word_windows,window),axis = 0)    
                    
    else:
        word_list = []
        word_windows = np.zeros((1,31))
        for s,t in zip(sentences,sentence_tags):
            if len(s)<3:
                pass
            else:
                for i in range(len(s)):
                    if s[i] not in word_list:
                        word_list.append(s[i])
                    if i == 0:
                        main_word = word2vec(s[0],model).reshape(1,-1)
                        prev_word = np.zeros((1,10))
                        next_word = word2vec(s[1],model).reshape(1,-1)
                        tag_id = tags.index(t[0])
                        window = np.concatenate((main_word,prev_word,next_word),axis = 1)
                        window = np.append(window,tag_id).reshape(1,-1)
                        
                    if i == len(s) -1 :
                        main_word = word2vec(s[-1],model).reshape(1,-1)
                        prev_word = word2vec(s[-2],model).reshape(1,-1)
                        next_word = np.zeros((1,10))
                        tag_id = tags.index(t[-1])
                        window = np.concatenate((main_word,prev_word,next_word),axis = 1)
                        window = np.append(window,tag_id).reshape(1,-1)
                    else:
                        main_word = word2vec(s[i],model).reshape(1,-1)
                        prev_word = word2vec(s[i-1],model).reshape(1,-1)
                        next_word = word2vec(s[i+1],model).reshape(1,-1)
                        tag_id = tags.index(t[i])
                        window = np.concatenate((main_word,prev_word,next_word),axis = 1)
                        window = np.append(window,tag_id).reshape(1,-1)
                    word_windows = np.concatenate((word_windows,window),axis = 0)
        word_windows = np.concatenate((word_windows,np.array(word_list).reshape(-1,1)),axis = 1)    
    #random.shuffle(word_windows)
    word_windows = word_windows[1:]
    return word_windows
 
def make_sparse_examples_(dense_examples):
    if not dense_examples:
        print('Please use make_dense_examples first!')
    examples = dense_examples
    x = examples[:,:-1]
    y_ = examples[:,-1].astype('int')
    ydim = np.max(y_)+1
    y = np.zeros((x.shape[0],ydim))
    y[np.arange(y_.shape[0]),y] = 1
    return x,y



def datadump (network,name):
    name = name +'_trained_weights.pkl'
    with open(name,'wb') as file:
        pickle.dump(network.weights,file)    
    name = name + '_trained_bias.pkl'
    with open(name,'wb') as file:
        pickle.dump(network.bias,file)
    name = name + '_loss_log.pkl'
    with open(name,'wb') as file:
        pickle.dump(network.loss_log,file)
    name = name + '_accuracy_log.pkl'
    with open(name,'wb') as file:
       pickle.dump(network.accuracy_log,file)   
    name = name + '_lr_log.pkl'   
    with open(name,'wb') as file:
        pickle.dump(network.lr_log,file)    
    name = name + '_val_loss_log'  
    with open(name,'wb') as file:
        pickle.dump(network.val_loss_log,file)  
    name = name + '_i_log.pkl'
    with open(name,'wb') as file:
        pickle.dump(network.i_log,file)    



#useless,sentence_tags_dev = get_tags(path_dev)
#tags,sentence_tags_train = get_tags(path_train)
#useless,sentence_tags_test = get_tags(path_test)

with open('sentences_train.pkl','rb') as file:
    sentences_train = pickle.load(file)
    
with open('sentences_test.pkl','rb') as file:
    sentences_test = pickle.load(file)
    
with open('sentences_dev.pkl','rb') as file:
    sentences_dev = pickle.load(file)
    

   
UNK_replace(sentences_train,sentence_tags_train)
UNK_replace(sentences_test,sentence_tags_test)
UNK_replace(sentences_dev,sentence_tags_dev)

#examples = make_dense_examples(sentences_train,sentence_tags_train,tags,model_10)
dev = make_dense_examples(sentences_dev,sentence_tags_dev,tags,model_10)
#test = make_dense_examples(sentences_test,sentence_tags_test,tags,model_10)

#random.shuffle(examples)
#with open('examples_train.pkl','wb') as file:
    #pickle.dump(examples,file)    

x_val,y_val = make_parse_examples_(dev)












      
        
