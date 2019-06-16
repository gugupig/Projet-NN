import numpy as np
# from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numba as nb
from numba import cuda
import time


class NN:
    def __init__(self, neurons):
        self.neurons = neurons
        self.n_layers = len(self.neurons)
        self.weights = []
        self.bias = []
        self.act = []
        self.z = []
        self.final_weights = []
        self.final_bias = []
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
            return np.zeros(5)
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
        if not self.final_weights or not self.final_bias:
            raise Exception('Please train first')
        else:
            return self.forward(self.weights, self.bias,x)

    def mini_batch_train(self, X,Y,batch_size,epoch,learning_rate):
        loss = 0
        loss_log = []
        i_log = []
        self.initialisation(X, Y)
        for i in range(epoch):
            if i%50 == 0 and i !=0:
                loss_log.append(loss)
                i_log.append(i)
                print('Total loss at epoch', i,loss)
            #for x, y in zip(t_X,t_Y):
            for i in range(0,X.shape[0],batch_size):
                slice_x = X[i:i+batch_size]
                slice_y = Y[i:i+batch_size]
                weights, bias = self.weights,self.bias
                y_pred = self.forward(weights, bias,slice_x)
                loss =  self.loss_func(slice_y, y_pred,'ford').sum()
                initial_grad = self.loss_func(slice_y, y_pred,'back')
                weights_grad, bias_grad = self.backward(initial_grad,weights,bias)
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] - learning_rate * 1/slice_x.shape[0]*weights_grad[j]
                for k in range(len(self.bias)):
                    self.bias[k] = self.bias[k] - learning_rate *1/slice_x.shape[0]*bias_grad[k]
            loss += 1/slice_x.shape[0] * loss
        self.final_weights = self.weights
        self.final_bias = self.bias


    def batch_train(self, X,Y,epoch,learning_rate):
        loss = 0
        loss_log = []
        i_log = []
        self.initialisation(X, Y)
        for i in range(epoch):
            if i%50 == 0 and i !=0:
                loss_log.append(loss)
                i_log.append(i)
                print('Total loss at epoch', i,loss)
            weights, bias = self.weights,self.bias
            y_pred = self.forward(weights, bias,X)
            loss =  self.loss_func(Y, y_pred,'ford').sum()
            initial_grad = self.loss_func(Y, y_pred,'back')
            weights_grad, bias_grad = self.backward(initial_grad,weights,bias)
            for j in range(len(self.weights)):
                self.weights[j] = self.weights[j] - learning_rate * 1/X.shape[0]*weights_grad[j]
            for k in range(len(self.bias)):
                self.bias[k] = self.bias[k] - learning_rate *1/X.shape[0]*bias_grad[k]
            loss += 1/X.shape[0] * loss
        self.final_weights = self.weights
        self.final_bias = self.bias


x = np.array([[1., 0.],[0.,1.],[1.,1.],[0.,0.],[1., 0.]])
y = np.array([[1., 0.],[1.,0.],[0.,1.],[0.,1.],[1., 0.]])

nn = NN([15])
#nn.mini_batch_train(x, y,5,1,0.1)
nn.batch_train(x, y,5000,0.1)
prd = nn.predit(x)
print(prd,y)