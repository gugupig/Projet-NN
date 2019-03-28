# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:37:38 2019

@author: Kurros
"""
import numpy as np


class NN:
    def __init__(self,neurons,activate,output_func,loss_func):
        self.neurons = neurons
        self.n_layers = len(self.neurons)
        self.weights = []
        self.bias = []
        if activate =='sign':
             self.activate = self.act_sign
        elif activate == 'tanh':
            self.activate = self.act_tanh
        elif activate == 'sigmoid':
            self.activate = self.act_sig
        elif activate == 'relu':
            self.activate = self.act_relu
        else:
            raise Exception ('Plase specify activation function')
        if output_func == 'softmax':
            self.output_func = self.softmax
        elif output_func == 'tanh':
            self.output_func = self.tanh
        elif output_func == 'sign':
            self.output_func = self.act_sign
        else:
            raise Exception('Please specify output funciton')
        if loss_func == 'sqrt':
            self.loss_func = self.loss_sqrt
        elif loss_func == 'cross':
            self.loss_func = self.loss_cross_entro
        else:
            raise Exception('Please specify loss funciton')
            
            
            
    def feed_forward(self,x):
        curr_input = x
        curr_output = x
        final_output = 0
        for w,b in zip(self.weights,self.bias):
            curr_input = np.dot(curr_output,w)+b
            #print(curr_input)
            curr_output =self.activate(curr_input)
            #print(curr_output)
        final_output = self.output_func(curr_output)
        return final_output
    
    def initial_weights(self,x,y):
        x_dim = len(x)
        y_dim = len(y)
        for layer in self.neurons:
            self.weights.append(np.random.rand(x_dim,layer))
            self.bias.append(np.random.rand(layer))
            x_dim = self.weights[-1].shape[1]
        self.weights.append(np.random.rand(self.neurons[-1],y_dim))
        self.bias.append(np.random.rand(y_dim))
    
    def act_sig(self,x,mode = 'ford'):
        result = np.array([])
        for i in x:
            result = np.append(result,(1/(np.exp(-i))))
        return result
    
    def act_sign(self,x):
        result = np.array([])
        for i in x:
            if i > 0:
                result = np.append(result,1)
            elif i < 0:
                result = np.append(result,-1)
            elif i == 0:
                raise Exception('0!!')
        return result
    
    def act_tanh(self,x,mode = 'ford'):
        pass
    
    def softmax(self,x,mode = 'ford'):
        pass
    
    def act_relu(self,x,mode = 'ford'):
        pass
    
    def loss_sqrt(self,y_glod,y_pred):
        pass
    def loss_cross_entro(self,y_glod,y_pred):
        pass
    
    
    def GD(self,x_0,grad_f,tau,n_iteration):
        x = [x_0]
        for i in range(n_iteration):
            x_new = x[-1] - tau*grad_f(x[-1])
        return x_new
        
        

xor = NN([1],'sign','sign','cross')
xor.weights = [np.array([[-1,1],[-1,1]]),np.array([1,1])]
xor.bias =[np.array([1,1]),np.array([-1])]
out = xor.feed_forward(np.array([-1,1]))
for xin in [np.array([1,1]),np.array([-1,-1]),np.array([1,-1]),np.array([-1,1])]:
    print(xin,int(xor.feed_forward(xin)[0]))
