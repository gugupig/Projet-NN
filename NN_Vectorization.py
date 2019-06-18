# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:41:54 2019

@author: Kurros LAN
"""

# -*- coding: utf-8 -*- 
""" 
Created on Tue Mar 26 10:37:38 2019 
 
@author: Lan 
""" 
 
 
import numpy as np 
# from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt 
import time 
 
 
#Vectorized version of NN class 
class NN_Vec: 
    def __init__(self, neurons): 
        self.neurons = neurons 
        self.n_layers = len(self.neurons) 
        self.weights = [] 
        self.bias = [] 
        self.act = [] 
        self.z = [] 
        self.i_log = [] 
        self.loss_log = [] 
        self.accuracy_log = [] 
        self.lr_log = [] 
        self.activate = self.act_sig 
        self.output_func = self.softmax 
        self.loss_func = self.loss_cross_entropy 
 
    def initialisation(self, x, y): 
        self.weights = [] 
        self.bias = [] 
        self.act = [] 
        self.weights.append(np.random.uniform(-1, 1, (x.shape[1], self.neurons[0]))) 
        self.bias.append(np.random.uniform(-1, 1, (1, self.neurons[0]))) 
        for layer in range(1,len(self.neurons)): 
            self.weights.append(np.random.uniform(-1, 1, (self.weights[-1].shape[1], self.neurons[layer]))) 
            self.bias.append(np.random.uniform(-1, 1, (1, self.neurons[layer]))) 
         
        self.weights.append(np.random.uniform(-1, 1, (self.neurons[-1], y.shape[1]))) 
        self.bias.append(np.random.uniform(-1, 1, (1, y.shape[1]))) 
 
    def forward(self, weights, bias, x): 
        curr_input = x 
        curr_output = x 
        self.act = [x] 
        self.z = [] 
        counter = 0 
        for w, b in zip(weights, bias): 
            curr_input = np.dot(curr_output,w) + b 
            self.z.append(curr_input) 
            counter += 1 
            if counter < len(weights): 
                curr_output = self.activate(curr_input) 
            else: 
                curr_output = self.output_func(curr_input) 
            self.act.append(curr_output) 
        return curr_output 
 
    def act_sig(self, x, mode='ford'): 
        if mode == 'ford': 
            result = 1. / (1. + np.exp(-x)) 
        elif mode == 'back': 
            result = self.act_sig(x) * (1 - self.act_sig(x)) 
            #result = 1. / (1. + np.exp(-1 * x)) * (1. - (1. / (1. + np.exp(-1 * x)))) 
        return result 
 
    def softmax(self, x, mode='ford'): 
        exp = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        if mode == 'ford': 
            return exp / exp.sum(axis=1, keepdims=True) 
        if mode == 'back': 
            ford = self.softmax(x, mode='ford') 
            step = ford.shape[1] 
            ford = ford.reshape(-1,1) 
            dim = ford.shape[0] 
            block = np.dot(ford[0:step],ford[0:step].T) 
            diag = np.diagflat(ford[0:step]) 
            block = diag - block 
            for i in range(step,dim,step): 
                sub = ford[i:i+step] 
                diag = np.diagflat(sub) 
                next_block = diag - np.dot(sub,sub.T) 
                block = np.concatenate((block,next_block)) 
            return block 
     
    def act_sign(self,x): 
        y = x.copy() 
        y[y < 0 ] = -1 
        y[y > 0 ] = 1 
        return y 
     
    def loss_cross_entropy(self, y_gold, y_pred, mode='ford'): 
 
        if mode == 'ford': 
            return np.sum((-1*y_gold)*np.log(y_pred)) 
        if mode == 'back': 
            with np.errstate(divide = 'ignore'): 
                nozeros = (-1 / (y_pred*y_gold)) 
                nozeros[nozeros == -np.inf] = 0 
            return nozeros 
     
    def multi_dot(self,dl_da,da_dz): 
        axe_x = dl_da.shape[0] 
        axe_y = dl_da.shape[1] 
        dlda = dl_da.reshape(axe_x,1,axe_y) 
        dadz = da_dz.reshape(axe_x,axe_y,axe_y) 
        dot = np.einsum('ijp,ipj->ij',dlda,dadz) 
        return dot 
             
     
     
    def backward(self, initial_grad, weights, bias): 
        dl_da = initial_grad 
        weights_grad = [] 
        bias_grad = [] 
        inverse_weights = weights[::-1] 
        inverse_act = self.act[::-1] 
        inverse_z = self.z[::-1] 
        for layer in range(len(weights)): 
            if layer == 0 : 
                da_dz = self.output_func(inverse_z[layer],'back') 
                dl_dz = self.multi_dot(dl_da,da_dz) 
            else: 
 
                da_dz = self.activate(inverse_z[layer], 'back') 
                dl_dz = (np.multiply(dl_da.T, da_dz)) 
            dl_da = np.dot(inverse_weights[layer],dl_dz.T) 
            weights_grad.append((np.dot(dl_dz.T, inverse_act[layer + 1])).T) 
            bias_grad.append(np.sum(dl_dz,axis =0,keepdims = True)) 
        return weights_grad[::-1], bias_grad[::-1] 
 
 
    def predit(self, x): 
        return self.forward(self.weights, self.bias,x) 
 
    def mini_batch_train(self,X,Y,batch_size,epoch,learning_rate = 0.1,decay = 0,beta = 0 ,x_val = 0 ,y_val = 0): 
        print('--Training begins--') 
        loss = 0 
        self.loss_log = [0] 
        self.i_log = [] 
        self.lr_log = [] 
        self.initialisation(X, Y) 
        v_w = [0 for i in range(len(self.weights))] 
        v_b = [0 for i in range(len(self.weights))] 
        for i in range(epoch): 
            if i%50 == 0 and i !=0: 
                if x_val != 0 and y_val != 0: 
                    self.accuracy_log(self.accuracy_test(x_val,y_val)) 
                    print(self.accuracy_log[-1]) 
                print('Diffrence:',loss - self.loss_log[-1]) 
                print('Total loss at epoch', i,loss) 
                self.loss_log.append(loss) 
                loss = 0 
                self.i_log.append(i) 
                if decay != 0: 
                    self.lr_log.append(learning_rate) 
            for i in range(0,X.shape[0],batch_size): 
                learning_rate = (1./(1. + decay*i))*learning_rate #using decay learning rate method 
                slice_x = X[i:i+batch_size] 
                slice_y = Y[i:i+batch_size] 
                weights, bias = self.weights,self.bias 
                y_pred = self.forward(weights, bias,slice_x) 
                batch_loss = 1/X.shape[0]*self.loss_func(slice_y, y_pred,'ford') 
                initial_grad = self.loss_func(slice_y, y_pred,'back') 
                weights_grad, bias_grad = self.backward(initial_grad,weights,bias) 
                for j in range(len(self.weights)): 
                    v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j]  
                    self.weights[j] = self.weights[j] - learning_rate * 1/slice_x.shape[0]*v_w[j] 
                for k in range(len(self.bias)): 
                    v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                    self.bias[k] = self.bias[k] - learning_rate *1/slice_x.shape[0]*v_b[k] 
                loss += batch_loss 
 
         
         
         
         
 
    def accuracy_test(self,X,Y,onehot = True,boundary = 0.9): 
        result = self.predit(X) 
        if onehot == True: 
            result[result >= boundary] = 1 
            good = np.sum(Y.reshape(result.shape[0],result.shape[1]) == result) 
        return float(good)/float(Y.shape[0]) 
 
    def batch_train(self,X,Y,epoch,learning_rate,beta = 0,x_val = 0,y_val = 0): 
        print('--Training begins--') 
        loss = 0 
        self.loss_log = [] 
        self.i_log = [] 
        self.initialisation(X, Y) 
        v_w = [0 for i in range(len(self.weights))] 
        v_b = [0 for i in range(len(self.weights))] 
        for i in range(epoch): 
            if i%50 == 0 and i !=0: 
                if x_val !=0 and y_val !=0: 
                    self.accuracy_log(self.accuracy_test(x_val,y_val)) 
                    print(self.accuracy_log[-1]) 
                self.loss_log.append(loss) 
                loss = 0 
                self.i_log.append(i) 
                print('Total loss at epoch', i,loss) 
            weights, bias = self.weights,self.bias 
            y_pred = self.forward(weights, bias,X) 
            loss =  self.loss_func(Y, y_pred,'ford').sum() 
            initial_grad = self.loss_func(Y, y_pred,'back') 
            weights_grad, bias_grad = self.backward(initial_grad,weights,bias) 
            for j in range(len(self.weights)): 
                v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j] 
                self.weights[j] = self.weights[j] - learning_rate * 1/X.shape[0]*weights_grad[j] 
            for k in range(len(self.bias)): 
                v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                self.bias[k] = self.bias[k] - learning_rate *1/X.shape[0]*bias_grad[k] 
            loss += 1/X.shape[0] * loss 
 
 
 
    def mini_batch_train_no_sparse(self,training_set,batch_size,epoch,learning_rate,beta = 0, validation = 0): 
        print('--Training begins--') 
        loss = 0 
        self.loss_log = [] 
        self.i_log = [] 
        x_train = training_set[:,:-1] 
        y_train = training_set[:,-1] 
        max_dim = int(np.max(y_train))+1 
        self.initialisation(x_train, np.zeros((1,max_dim))) 
        v_w = [0 for i in range(len(self.weights))] 
        v_b = [0 for i in range(len(self.weights))] 
        for i in range(epoch): 
            if i%50 == 0 and i !=0: 
                if validation != 0: 
                    x_val = validation[:,:-1] 
                    y_val = validation[:,-1] 
                    self.accuracy_log.append(self.accuracy_test(x_val,y_val)) 
                    print(self.accuracy_log[-1]) 
                self.loss_log.append(loss) 
                self.i_log.append(i) 
                print('Total loss at epoch', i,loss) 
            for i in range(0,x_train.shape[0],batch_size): 
                slice_x = x_train[i:i+batch_size] 
                index = y_train[i:i+batch_size].astype('int') 
                slice_y = np.zeros((index.shape[0],max_dim)) 
                slice_y[np.arange(index.shape[0]),index] = 1. 
                weights, bias = self.weights,self.bias 
                y_pred = self.forward(weights, bias,slice_x) 
                batch_loss = 1/slice_x.shape[0] * self.loss_func(slice_y, y_pred,'ford') 
                initial_grad = self.loss_func(slice_y, y_pred,'back') 
                weights_grad, bias_grad = self.backward(initial_grad,weights,bias) 
                for j in range(len(self.weights)): 
                    v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j]  
                    self.weights[j] = self.weights[j] - learning_rate * 1/slice_x.shape[0]*v_w[j] 
                for k in range(len(self.bias)): 
                    v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                    self.bias[k] = self.bias[k] - learning_rate *1/slice_x.shape[0]*v_b[k] 
                loss +=  batch_loss 
 
    def input_parameters(self,weights,bias): 
        self.weights = weights 
        self.bias = bias 
         
 
    def learning_rate_finder(self,X,Y,beta = 0.8,initial_rate = 1e-6,multiplier = 1.1,stop = 10000,sample_size = 50): 
        print('--Training begins--') 
        self.initialisation(X,Y) 
        loss_log = [] 
        learning_rate_log = [] 
        v_w = [0 for i in range(len(self.weights))] 
        v_b = [0 for i in range(len(self.weights))] 
        learning_rate = initial_rate 
        runs = 0 
        while True:       
            sample_idxs = np.unique(np.random.randint(X.shape[0], size= sample_size)) 
            x_samples = X[sample_idxs] 
            y_samples = Y[sample_idxs] 
            weights, bias = self.weights,self.bias 
            y_pred = self.forward(weights, bias,x_samples) 
            batch_loss = 1/X.shape[0]*self.loss_func(y_samples, y_pred,'ford') 
            initial_grad = self.loss_func(y_samples, y_pred,'back') 
            weights_grad, bias_grad = self.backward(initial_grad,weights,bias) 
            for j in range(len(self.weights)): 
                v_w[j] = beta*v_w[j] + (1-beta)*weights_grad[j]  
                self.weights[j] = self.weights[j] - learning_rate * 1/x_samples.shape[0]*v_w[j] 
            for k in range(len(self.bias)): 
                v_b[k] = beta*v_b[k] + (1-beta)*bias_grad[k] 
                self.bias[k] = self.bias[k] - learning_rate *1/x_samples.shape[0]*v_b[k] 
            learning_rate = learning_rate * multiplier 
            learning_rate_log.append(learning_rate) 
            loss_log.append(batch_loss) 
            runs +=1 
            stop_condition = (sum(loss_log)/float(len(loss_log))) > 2 * loss_log[0] 
            if (stop_condition and len(loss_log) > 20) or runs>stop: 
                return learning_rate_log,loss_log 
             
 
 
'''   
 
x = np.array([[1., 0.],[0.,1.],[1.,1.],[0.,0.],[1., 0.]]) 
y = np.array([[1., 0.],[1.,0.],[0.,1.],[0.,1.],[1., 0.]]) 
 
st = time.time() 
nn = NN_Vec([15]) 
nn.mini_batch_train(x, y,5,2000,0.1) 
#nn.batch_train(x, y,5000,0.1) 
prd = nn.predit(x) 
print(prd,y) 
ed = time.time() 
print(ed-st) 
'''