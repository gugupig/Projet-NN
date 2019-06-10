    # -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:37:38 2019

@author: Kurros
"""
import numpy as np
#from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


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
        else:
            raise Exception('Please specify loss funciton')
            
            
    def vector_multiply(self,a,b):
        if a.shape == ():
            multiply = np.zeros((1,b.shape[0]))
            for i in range(1):
                for j in range(b.shape[0]):
                    multiply[i][j] = a*b[j]
        else:
            multiply = np.zeros((a.shape[0],b.shape[0]))
            for i in range(a.shape[0]):
                for j in range(b.shape[0]):
                    multiply[i][j] = a[i]*b[j]
        return multiply
            
    def feed_forward(self,weights,bias,x):
        curr_input = x
        curr_output = x
        self.act = [x]
        self.z = []
        counter = 0
        for w,b in zip(weights,bias):
            curr_input = np.dot(curr_output,w)+b
            self.z.append(curr_input)
            counter += 1
            if counter < len(weights):    
                curr_output =self.activate(curr_input)
            else:
                curr_output = self.output_func(curr_input)
            self.act.append(curr_output)
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
                tanh = (1.0 - np.exp(-2*i))/(1.0 + np.exp(-2*i))
                result = np.append(result,tanh)
        elif mode == 'back':
            for i in x :
                tanh = (1.0 - np.exp(-2*i))/(1.0 + np.exp(-2*i))
                result = np.append(result,(1 + tanh)*(1 - tanh))
        return result
    
    def softmax(self,x,mode = 'ford',onehot = True):
        ex = np.exp(x- np.max(x))
        if mode == 'ford':
            return ex/ex.sum()
        if mode == 'back':
            ford = self.softmax(x,mode = 'ford')
            jacobian = np.zeros((ford.size,ford.size))
            for i in range(len(jacobian)):
                for j in range(len(jacobian)):
                    if i == j:
                        jacobian[i][j] = ford[i]*(1-ford[j])
                    else:
                        jacobian[i][j] = -1*ford[i]*ford[j]
        return jacobian
    
    def act_relu(self,x,mode = 'ford'):
        pass
    
    """
    def loss_cross_entro(self,y_gold,y_pred,mode = 'ford'):
        index_gold = np.nonzero(y_gold)[0][0]    #one-hot version
        if mode == 'ford':
            loss = -1*(np.log(y_pred[index_gold]))
            return loss
        if mode == 'back':
            return -1/y_pred[index_gold]

    """
    def loss_cross_entro(self,y_gold,y_pred,mode = 'ford'):
        index_gold = np.nonzero(y_gold)[0][0]
        dot_product = np.dot(-1*y_gold,np.log(y_pred))
        if mode == 'ford':
            return dot_product
        if mode == 'back':
            derive = np.zeros(y_pred.shape)
            derive[index_gold] = -1/y_pred[index_gold]
            return derive
    
    #需要修改成向量版本
    def loss_sqrt(self,y_gold,y_pred,mode = 'ford'):
        if mode == 'ford':
            return 0.5*(y_gold-y_pred)**2
        elif mode == 'back':
            return -(y_gold-y_pred)        
    
    def act_test(self,x):
        return x
        
    
    def GD(self,x_0,grad_f,tau,n_iteration):
        x = [x_0]
        for i in range(n_iteration):
            x_new = x[-1] - tau*grad_f(x[-1])
        return x_new
        
    def backward(self,initial_grad,weights,bias):
        dl_da = initial_grad
        weight_grad = []
        bias_grad = []
        inverse_weights = weights[::-1]
        inverse_act = self.act[::-1]
        inverse_z = self.z[::-1]
        for i in range(len(inverse_weights)):
            #需要修改适应非向量（2，1）无法与(2,)dot product
            if i == 0:
                #dl_dz = np.multiply(dl_da , self.output_func(inverse_z[i],'back'))
                dl_dz = np.dot(dl_da,self.output_func(inverse_z[i],'back'))
            else:
                #print(dl_da.shape)
                #print(self.activate(inverse_z[i],'back').shape)
                dl_dz = np.dot(dl_da,self.activate(inverse_z[i],'back'))
                #dl_dz = np.dot(dl_da,self.activate(inverse_z[i],'back').reshape(dl_da.shape).T)
                #dl_dz = np.multiply(dl_da , self.activate(inverse_z[i],'back'))
            dl_da = np.dot(inverse_weights[i],dl_dz)        
            weight_grad.append((self.vector_multiply(dl_dz.T,inverse_act[i+1])).T)      
            bias_grad.append(dl_dz)
        return weight_grad[::-1],bias_grad[::-1]
    


    def train(self,examples,batchsize = 4,epoch = 1,learning_rate =0.1):
        loss = 0
        loss_log = []
        i_log = []
        self.initialisation(examples[0][0],examples[0][1])  
        for i in range (epoch):
            if i%1000 == 0 and i!=0:
                loss_log.append(loss)
                i_log.append(i)
                print ('Total loss at epoch',i, loss)
            weights,bias = self.weights,self.bias #update the weights and bias after one epoch
            for example in examples[:batchsize]:
                y_pred = self.feed_forward(weights,bias,example[0]) #use the same weights and bias to calculate )
                loss = self.loss_func(example[1],y_pred,'ford')
                initial_grad = self.loss_func(example[1],y_pred,'back')
                weight_grad,bias_grad = self.backward(initial_grad,weights,bias)
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] -  learning_rate* (1./batchsize)* weight_grad[j]             
                for k in range(len(self.bias)):
                    self.bias[k] = self.bias[k] - learning_rate* (1./batchsize)* bias_grad[k]
                loss +=(1./batchsize)*loss
        plt.ylabel('Total Loss')
        plt.xlabel('Epoch')
        plt.plot(i_log,loss_log)
        plt.show()
        self.final_weights = self.weights
        self.final_bias = self.bias
        
        
        
    def predit(self,x):
        if not self.final_weights or not self.final_bias:
            raise Exception('Please train first')
        else:
            return self.feed_forward(self.final_weights,self.final_bias,x)

"""
xor = NN([2],'sign','sign','cross')
xor.weights = [np.array([[-1,1],[-1,1]]),np.array([1,1])]
xor.bias =[np.array([1,1]),np.array([-1])]
out = xor.feed_forward(xor.weights,xor.bias,np.array([-1,1]))
for xin in [np.array([1,1]),np.array([-1,-1]),np.array([1,-1]),np.array([-1,1])]:
    print(xin,int(xor.feed_forward(xor.weights,xor.bias,xin)[0]))

xor_1 = NN([2],'tanh','tanh','sqrt')
exs_1 = [(np.array([1,1]),0),(np.array([0,0]),0),(np.array([1,0]),1),(np.array([0,1]),1)]
xor_1.train(exs_1,4,50000,0.08)
for xin in [np.array([1,1]),np.array([0,0]),np.array([1,0]),np.array([0,1])]:
    predit = xor_1.predit(xin)
    print('XOR_PREDIT:',xin,predit)


  
xor_2 = NN([2],'tanh','tanh','sqrt')
exs_2 = [(np.array([1,1]),-1),(np.array([-1,-1]),-1),(np.array([1,-1]),1),(np.array([-1,1]),1)]
xor_2.train(exs_2,4,50000,0.1)
for xin in [np.array([1,1]),np.array([-1,-1]),np.array([1,-1]),np.array([-1,1])]:
    predit = xor_2.predit(xin)
    print('XOR_PREDIT2:',xin,predit)
"""
xor_3 = NN([10],'sigmoid','softmax','cross')
exs_3 = [(np.array([1,1]),np.array([0,1])),(np.array([-1,-1]),np.array([0,1])),(np.array([1,-1]),np.array([1,0])),(np.array([-1,1]),np.array([1,0]))]
xor_3.train(exs_3,4,10000,0.1)
for xin in [np.array([1,1]),np.array([-1,-1]),np.array([1,-1]),np.array([-1,1])]:
    predit = xor_3.predit(xin)
    print('XOR_PREDIT3:',xin,predit)


xor_4 = NN([10],'sigmoid','softmax','cross')
exs_4 = [(np.array([1,1]),np.array([0,1])),(np.array([0,0]),np.array([0,1])),(np.array([1,0]),np.array([1,0])),(np.array([0,1]),np.array([1,0]))]
xor_4.train(exs_4,4,10000,0.08)
for xin in [np.array([1,1]),np.array([0,0]),np.array([1,0]),np.array([0,1])]:
    predit = xor_4.predit(xin)
    print('XOR_PREDIT4:',xin,predit)
 
