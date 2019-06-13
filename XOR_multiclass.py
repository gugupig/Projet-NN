import numpy as np

class NN:
	def __init__(self,neurons,activate,output_func,loss_func):
		# -neurons: is a list of number of neural units for every layer 
		# ex: neurons = [2,50,3],it's a two-layer network, the first layer
		# is assumed to be an input layer,containing 2 neurons, the second layer is hidden layer,
		# the third is output layer with 1 neuron.
		#-activate: activation function
		self.neurons = neurons
		self.n_layers = len(neurons)
		#The biases and weights are initialized randomly, 
		#using a Gaussian distribution with mean 0 and variance 1.
		self.weights = [np.random.randn(x, y) # ex: [w1.shape=(2,50),w2.shape=(50,3)]
						for x, y in zip(neurons[:-1], neurons[1:])]
		self.biases = [np.random.randn(1, y) for y in neurons[1:]]# ex: [b1.shape=(1,50),b1.shape(1,3)]

		self.a = [] # list to store all the activations, layer by layer
		self.z = [] # list to store all the pre-activations vectors, layer by layer
		

		if activate == 'tanh':
			self.activate = self.act_tanh
		elif activate == 'relu':
			self.activate = self.act_relu
		else:
			raise Exception ('Plase specify activation function')

		if output_func == 'softmax':
			self.output_func = self.softmax
		else:
			raise Exception('Please specify output funciton')

		if loss_func == 'sqrt':
			self.loss_func = self.loss_sqrt
		elif loss_func == 'cross':
			self.loss_func = self.loss_cross_entro
		else:
			raise Exception('Please specify loss funciton')

	def softmax(self,z,mode = 'ford'):
		if mode == 'ford':

			exp_scores = np.exp(z)
			probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
			return probs
		elif mode == 'back':
			#compute and return the loss gradient at output pre-activation

			# the output layer: dl_dz = a, except y_pred of the gold class
			dl_dz= self.a[-1]
			# dl_dz of y_pred of the gold class = a-1
			dl_dz[range(batchsize),y_gold]-=1

			return dl_dz

	def act_relu(self,z,mode = 'ford'):#???? l的值
		
		if mode == 'ford': # the relu activate fonction
			return np.maximum(0,z)

		elif mode == 'back': # derivatives of the ReLU fonction
			# vector of partials derivatives of the ReLU fonction has the same shape as the z at the same layer
			da_dz= z
			# if z>0, the partial derivative da/dz= 1
			da_dz[z>0] = 1
			# if z<=0, the partial derivative da/dz = 0
			da_dz[z<=0] = 0
			# retourne gradient of ReLU fonction 
			return da_dz

	def loss_cross_entro(self,y_gold,batchsize):
		"""Return the average data loss of a minibatch"""
		y_pred = self.a[-1]
		gold_logprobs = -np.log(y_pred[range(batchsize),y_gold])
		data_loss = np.sum(gold_logprobs)/batchsize
		
		return data_loss

	def feed_forward(self,X):
		"""
		-argument X: observations
		-compute and stock all the z, a values in self.a and self.z

		"""
		#the first layer:input layer 
		curr_input = X
		curr_output = X
		self.a.append(X)
		counter = 0
		for w,b in zip(self.weights,self.biases):
			curr_input = np.dot(curr_output,w)+b
			self.z.append(curr_input)
			counter += 1
			#if not the last layer
			if counter < len(self.weights):
				curr_output = self.activate(curr_input)
			#if it's the last layer(output layer)
			else:
				curr_output = self.output_func(curr_input)
			self.a.append(curr_output)
		
		return curr_output

	def backword(self,batchsize):

		# weight_grad and bias_grad are layer-by-layer lists of numpy arrays, similar to self.biaises
		weight_grad = [np.zeros(w.shape) for w in self.weights]
		bias_grad =  [np.zeros(b.shape) for b in self.biases]

		#loss gradient at output pre-activation
		delta = self.output_func(self.a,mode ='back')

		# back prograpation, compute the loss gradient of the last layer of parameters
		# dl_db = dl_dz•dz_db = dl_dz•1
		bias_grad[-1] = np.sum(delta,axis=0,keepdims=True)/batchsize
		
		# activations of the last hidden layer
		a = self.a[-2]
		
		#dl_dw = dl_dz•dz_dw = dl_dz• a
		weight_grad[-1] = np.dot(a.transpose(), delta)/batchsize

		# the variable l in the loop below means:
		# l= 1 the last layer of neurons;
		# l= 2 the second-last layer, and so on; python can use negative indices in lists

		for l in range(2,len(self.weights)+1):
			z = self.z[-l]
			#a = self.a[-l]
			da_dz = self.activate(z,mode = 'back')
			# dL_dz(l) = delta•dz(l+1)_da(l)*da(l)_dz(l) #element-wise product: *
			delta = np.dot(delta,self.weights[-l+1].T) * da_dz
			# dL_db(l) = dL_dz(l)•dz_db(l) = dl_dz(l)•1
			bias_grad[-l] = np.sum(delta,axis=0,keepdims=True)/batchsize
			#dL_dw(l) = dL_dz(l)•dz(l)_dw(l) = dL_dz(l)• a(l-1)
			s =self.a[-l-1]
			#print('s',type(s))
			#print(np.array(s).T)
			#print(s.T)
			weight_grad[-l] = np.dot(self.a[-l-1].transpose(),delta)/batchsize

		return (weight_grad,bias_grad) # a tuple of two list objets

		def train(self,X,batchsize=4,epoch=1,learning_rate = 0.1):
			






X = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
y_gold = np.array([0,1,1,0])
batchsize = 4

xor = NN([2,50,2],'relu','softmax','cross')
y_pred = xor.feed_forward(X)
loss = xor.loss_cross_entro(y_gold,batchsize)
print(loss)

print(y_pred)
#print(xor.a)

#print(xor.weights,xor.biases)
g = xor.backword(batchsize)
print(g)


# 获取训练样本数量
#num_examples = X.shape[0]