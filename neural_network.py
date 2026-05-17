import numpy
import random
import scipy.special

class neuralNetwork:
	
	def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
		"""
		initialize the network
		"""
		self.inodes=inputnodes
		self.hnodes=hiddennodes
		self.onodes=outputnodes
		
		#learning rate
		self.lr=learningrate
		
		#activation  function sigmoid
		self.activation_function=lambda x: scipy.special.expit(x)
		
		# the weights of the network 00 is for picking normal probality distribution
		#weight bettwen input and hidden node
		self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
		
		#weight bettwen hidden and output node
		self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))		
		

	def train(self,inputs_list,targets_list):
		"""
		make the network ltrain
		"""
		#convert the inputs_list to 2d array
		inputs=numpy.array(inputs_list, ndmin=2).T
		targets=numpy.array(targets_list,ndmin=2).T
		# calculate the input to hidden nodes
		hidden_inputs=numpy.dot(self.wih,inputs)
		#calculate output from hidden nodes
		hidden_outputs=self.activation_function(hidden_inputs)
		#calculate input for output nodes
		final_inputs=numpy.dot(self.who,hidden_outputs)
		#calculate output
		final_outputs=self.activation_function(final_inputs)
		#error=(target-actual)
		output_errors= (targets-final_outputs)
		hidden_errors=numpy.dot(self.who.T,output_errors)
		
		#update the weight between hidden and output layer
		self.who += self.lr*numpy.dot((output_errors*final_outputs *(1.0-final_outputs)),numpy.transpose(hidden_outputs))
		#update the weight between input and hidden layer
		self.wih +=self.lr*numpy.dot((hidden_errors*hidden_outputs *(1.0-hidden_outputs)),numpy.transpose(inputs))
		
				
	def query(self,inputs_list):
		"""
		query the network(output)
		"""
		#convert the inputs_list to 2d array
		inputs=numpy.array(inputs_list, ndmin=2).T
		# calculate the input to hidden nodes
		hidden_inputs=numpy.dot(self.wih,inputs)
		#calculate output from hidden nodes
		hidden_outputs=self.activation_function(hidden_inputs)
		#calculate input for output nodes
		final_inputs=numpy.dot(self.who,hidden_outputs)
		#calculate output
		final_outputs=self.activation_function(final_inputs)
		return final_outputs
		

#set the number of nodes in each layer
input_nodes=784
hidden_nodes=200
output_nodes=10
#learning rate is 0.1
learning_rate=0.1
	
heisenberg=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#load the mnist data set

with open('mnist_train.csv') as f:
	traning_list=f.readlines()

#train the neural network
# epochs is the number of times the training data set is used for training
epochs = 5
for e in range(epochs):
	print('traning')
	#go through all the data
	for line in traning_list: 
	#split the lines by comma
		all_values=line.split(',')
		# scale it between 0.01 an 0.99
		inputs=numpy.asarray(all_values[1:],dtype=float)
		inputs=((inputs/255)*0.99)+0.01
		# create a target array set everything to 0.01 except the target which is 0.99
		targets=numpy.zeros(output_nodes)+0.01
		#all_values[0] 0] is the target label for this record
		targets[int(all_values[0])]=0.99
		heisenberg.train(inputs,targets)
score=[]
#test
#load the test mnist
with open('mnist_test.csv') as f:
	n=f.readlines()
#go through the data
for i in n:
	line=i.split(',')
	#change to array without the lebel or the first element
	input=numpy.asarray(line[1:],dtype=float)
	#scaling
	input=((input/255)*0.99)+0.01
	neuron=numpy.argmax(heisenberg.query(input))
	if neuron==int(line[0]):
		score.append(1)
	else:
		score.append(0)
print((sum(score)/len(score))*100)
import pickle

# Save weights
with open('wih.pkl', 'wb') as f:
    pickle.dump(heisenberg.wih, f)

with open('who.pkl', 'wb') as f:
    pickle.dump(heisenberg.who, f)

# Load weights
#with open('wih.pkl', 'rb') as f:
#    heisenberg.wih = pickle.load(f)

#with open('who.pkl', 'rb') as f:
#    heisenberg.who = pickle.load(f)