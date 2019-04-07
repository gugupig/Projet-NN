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
        self.act = []
        self.z = []
        self.final_weights = []
        self.final_bias = []
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
            self.output_func = self.act_tanh
        elif output_func == 'sign':
            self.output_func = self.act_sign
        elif output_func == 'sigmoid':
            self.output_func = self.act_sig
        else:
            raise Exception('Please specify output funciton')
        if loss_func == 'sqrt':
            self.loss_func = self.loss_sqrt
        elif loss_func == 'cross':
            self.loss_func = self.loss_cross_entro
        elif loss_func == 'sub':
            self.loss_func = self.loss_substraction
        else:
            raise Exception('Please specify loss funciton')
            
    def matrix_multiply(self,a,b):
        multiply = np.zeros((a.shape[0],b.shape[0]))
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                multiply[i][j] = a[i]*b[j]
        return multiply
            
    def feed_forward(self,weights,bias,x):
        curr_input = x
        curr_output = x
        #final_output = 0
        self.act = [x]
        self.z = []
        counter = 0
        for w,b in zip(weights,bias):
            curr_input = np.dot(curr_output,w)+b
            self.z.append(curr_input)
            counter += 1
            #print(curr_input)
            if counter < len(weights):    
                curr_output =self.activate(curr_input)
            else:
                curr_output = self.output_func(curr_input)
            #print(curr_output)
            self.act.append(curr_output)
        #final_output = self.output_func(curr_output)
        #self.act.append(final_output)
        return curr_output
    
    def initialisation(self,x,y):
        self.weights = []
        self.bias = []
        self.act = []
        x_dim = len(x)
        try:
            y_dim = len(y)
        except:
            y_dim = 1
            
        for layer in self.neurons:
            self.weights.append(np.random.rand(x_dim,layer))
            self.bias.append(np.random.rand(layer))
            x_dim = self.weights[-1].shape[1]
        self.weights.append(np.random.rand(self.neurons[-1],y_dim))
        self.bias.append(np.random.rand(y_dim))
    
    def act_sig(self,x,mode = 'ford'):
        result = np.array([])
        if mode == 'ford':            
            for i in x:
                result = np.append(result,(1./(1.+np.exp(-i))))
        elif mode == 'back':
            for i in x:
                result = np.append(result,((1./(1.+ np.exp(-i)))*(1.-(1./(1.+np.exp(-i))))))
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
        result = np.array([])
        if mode == 'ford':
            for i in x:
                tanh = 1.0 - np.exp(-2*i)/(1.0 + np.exp(-2*i))
                result = np.append(result,tanh)
        elif mode == 'back':
            for i in x :
                tanh = 1.0 - np.exp(-2*i)/(1.0 + np.exp(-2*i))
                result = np.append(result,(1 + tanh)*(1 - tanh))
        return result
    
    def softmax(self,x,mode = 'ford'):
        pass
    
    def act_relu(self,x,mode = 'ford'):
        pass
    
    def loss_sqrt(self,y_gold,y_pred,mode = 'ford'):
        if mode == 'ford':
            return (y_gold-y_pred)**2
        elif mode == 'back':
            return 2.*(y_gold-y_pred)
        
    def loss_cross_entro(self,y_glod,y_pred,mode = 'ford'):
        pass
    
    def loss_substraction(self,y_gold,y_pred,mode = 'ford'):
        if mode == 'ford':
            return y_gold-y_pred
        elif mode == 'back':
            return -1.*y_pred
        
    
    def GD(self,x_0,grad_f,tau,n_iteration):
        x = [x_0]
        for i in range(n_iteration):
            x_new = x[-1] - tau*grad_f(x[-1])
        return x_new
        
    def backward(self,initial_grad,weights,bias):
        print('-------BACK BEGINS--------')
        dl_da = initial_grad
        weight_grad = []
        bias_grad = []
        inverse_weights = weights[::-1]
        inverse_act = self.act[::-1]
        inverse_bias = bias[::-1]
        inverse_z = self.z[::-1]
        c = 0
        print('weights',weights)
        print('bias',bias)
        print('--------LOOP BEGINGS!--------')
        for i in range(len(inverse_weights)):
            print('LOOP = ' ,c)
            print('============')
            print('invers_z = np.dot(inverse_act[i+1],inverse_weights[i]) + inverse_bias[i])')
            print('inverse_weights=',inverse_weights[i],)
            print('inverse_act=',inverse_act[i+1])
            print('inverse_bias=',inverse_bias[i],)
            print('dotproduct = ',np.dot(inverse_act[i+1],inverse_weights[i]))
            #inverse_z = np.dot(inverse_act[i+1],inverse_weights[i]) + inverse_bias[i]
            print('inver_z = ',inverse_z[i])
            #print('2nd IZ :',np.dot(inverse_act[i+1],inverse_weights[i]) + inverse_bias[i])
            print('============')
            print("dl_dz = dl_da*act")
            print ('dl_da =',dl_da)
            print ('act = ',self.activate(inverse_z[i],'back'))
            dl_dz = np.multiply(dl_da , self.activate(inverse_z[i],'back'))
            print('dl_dz = ',dl_dz)
            print('=============')
            print('dl_da = dot (dl_dz,inverse_weights[i].T)')
            print('dl_dz = ',dl_dz)
            print('inverse_weight = ',inverse_weights[i].T)
            if len(dl_dz)==1:
                dl_da = np.multiply(dl_dz,inverse_weights[i].T)
            else:
                dl_da = np.dot(inverse_weights[i].T,dl_dz)
            print ('dl_da = ',dl_da)
            print('=============')
            print('new weight_grad  = inva.T * dl_dz')
            print('inva.T = ',inverse_act[i+1].T,inverse_act[i+1].T.shape)
            print('dl_dz = ',dl_dz,dl_dz.shape)
            #print('new grad = ',self.matrix_multiply(dl_dz.T,inverse_act[i+1]))
            #weight_grad.append((self.matrix_multiply(dl_dz.T,inverse_act[i+1])).T)
            weight_grad.append(np.dot(dl_dz,inverse_act[i+1].T))
            #print('new weight_grad = ',np.dot(dl_dz,inverse_act[i+1]))
            #weight_grad.append(np.dot(dl_dz,inverse_act[i+1])/inverse_act[i+1].shape[0])
            print('weight_grad_matrix = ',weight_grad[::-1])
            print('new bias = ',dl_dz)
            
            bias_grad.append(dl_dz)
            print('bias_grad_matrix =',bias_grad[::-1])
            c +=1
        print('---------------------BACK ENDS-------------------------')
        return weight_grad[::-1],bias_grad[::-1]
    


    def train(self,examples,batchsize = 1,epoch = 1,learning_rate =1):
        total_loss = 0
        self.initialisation(examples[0][0],examples[0][1])
        print('initial:',self.weights)
        for i in range (epoch):
            weights,bias = self.weights,self.bias #update the weights and bias after one epoch
            print('initial2:',self.weights)
            for example in examples[:batchsize]:
                y_pred = self.feed_forward(weights,bias,example[0]) #use the same weights and bias to calculate 
                print('YP:',y_pred)
                loss = self.loss_func(example[1],y_pred,'ford')
                print('LOSS',loss)
                initial_grad = self.loss_func(example[1],y_pred,'back')
                print('IG',initial_grad)
                weight_grad,bias_grad = self.backward(initial_grad,weights,bias)
                for j in range(len(self.weights)):
                    print('@@@@@@@@@@@@@@@@@@@@@@@@')
                    print("before:",self.weights)
                    print('============')
                    print('weight:',self.weights[j],self.weights[j].shape)
                    print('grad:',learning_rate* (1./batchsize)* weight_grad[j],(learning_rate* (1./batchsize)* weight_grad[j]).shape)
                    print('W-G:',self.weights[j] -  learning_rate* (1./batchsize)* weight_grad[j])
                    print('============')
                    self.weights[j] = self.weights[j] -  learning_rate* (1./batchsize)* weight_grad[j]
                    print("after:",self.weights)
                for k in range(len(self.bias)):
                    self.bias[k] = self.bias[k] - learning_rate* (1./batchsize)* bias_grad[k]
            total_loss = total_loss + (1./batchsize)*loss
            print('*****************************************')
            print ('TOTAL LOSS:',total_loss)
        self.final_weights = self.weights
        self.final_bias = self.bias
        
        
        
    def predit(self,x):
        if not self.final_weights or not self.final_bias:
            raise Exception('Please train first')
        else:
            return self.feed_forward(self.final_weights,self.final_bias,x)

xor = NN([2],'sign','sign','cross')
xor.weights = [np.array([[-1,1],[-1,1]]),np.array([1,1])]
xor.bias =[np.array([1,1]),np.array([-1])]
out = xor.feed_forward(xor.weights,xor.bias,np.array([-1,1]))
for xin in [np.array([1,1]),np.array([-1,-1]),np.array([1,-1]),np.array([-1,1])]:
    print(xin,int(xor.feed_forward(xor.weights,xor.bias,xin)[0]))

xor_1 = NN([2],'tanh','tanh','sub')
exs = [(np.array([1,1]),-1),(np.array([-1,-1]),-1),(np.array([1,-1]),1),(np.array([-1,1]),1)]
exs_1 = [(np.array([1,1]),0),(np.array([0,0]),0),(np.array([1,0]),1),(np.array([0,1]),1)]
xor_1.train(exs_1,4,1,1)
for xin in [np.array([1,1]),np.array([0,0]),np.array([1,0]),np.array([0,1])]:
    predit = xor_1.predit(xin)
    print(predit)



