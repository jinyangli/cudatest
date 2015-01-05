#!/usr/bin/python

import cPickle
import gzip
import numpy 

class LogisticLayer(object):
	def __init__(self, n_in, n_out, layer_no=0, totallayer=1):
		self.layer_no = layer_no
		self.totallayer = totallayer
		assert self.layer_no == (self.totallayer-1)
		self.W = numpy.zeros((n_in+1, n_out))

	def fprop(self, input):
		self.fprop_in = numpy.append(input, numpy.ones(shape=(input.shape[0],1)), 1)

		pred = numpy.exp(numpy.dot(self.fprop_in,self.W))
		norm = numpy.sum(pred,axis=1)
		pred = pred/ norm[:,numpy.newaxis]
		return pred

	def bprop_update(self, error,learning_rate):
		self.W -= learning_rate * numpy.dot(numpy.transpose(self.fprop_in), error)
		err = (1 - self.fprop_in * self.fprop_in) * numpy.dot(error, numpy.transpose(self.W))
		return err


class HiddenLayer(object):
	def __init__(self, n_in, n_out, layer_no=0, totallayer=1):
		self.layer_no = layer_no
		self.totallayer = totallayer
		self.W = numpy.asarray(numpy.random.RandomState(1234).uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in+1, n_out)), dtype=float)
		self.W *= 4
		# XXX: the bias is not zero initialized, don't care for now

	def fprop(self,input):
 		if self.layer_no == 0:
 			self.fprop_in = input
 		else:
 			self.fprop_in = numpy.tanh(input)
 			self.fprop_in = numpy.append(self.fprop_in, numpy.ones(shape=(self.fprop_in.shape[0],1)), 1)
 		return numpy.tanh(numpy.dot(self.fprop_in, self.W))

	def bprop_update(self, error, learning_rate):
		#error is n_out x 1 
		#output the propagated error
		if self.layer_no != 0:
			err = (1 - self.fprop_in * self.fprop_in) * numpy.dot(error[:,0:self.W.shape[1]], numpy.transpose(self.W))
		else:
			err = numpy.zeros(self.fprop_in.shape)
		self.W -= learning_rate * numpy.dot(numpy.transpose(self.fprop_in), error[:,0:self.W.shape[1]])
		return err

class NeuralNet(object):
	def __init__(self, units_array):
		self.layers = []
		sz = len(units_array)
		assert sz > 1
		for i in range(sz-2):
			self.layers.append(HiddenLayer(n_in=units_array[i], n_out=units_array[i+1], layer_no=i, totallayer=sz))
		self.layers.append(LogisticLayer(n_in=units_array[sz-2],n_out=units_array[sz-1],layer_no=sz-1,totallayer=sz))

	def update(self, input, output, label, learning_rate):
#	x = input
#		for i in range(len(self.layers)):
#			x = self.layers[i].fprop(x) 
		x = self.layers[1].fprop(self.layers[0].fprop(input))
		#correct_cnt = numpy.count_nonzero(label == numpy.argmax(x, axis=1))
		#print "%d out of %d correct" % (correct_cnt, output.shape[0])
		y = x - output
		for i in reversed(range(len(self.layers))):
			y = self.layers[i].bprop_update(error=y,learning_rate=learning_rate)

	def validate(self,input,output):
		x = input
		for i in range(len(self.layers)):
			x = self.layers[i].fprop(x)
		correct_cnt = numpy.count_nonzero(output== numpy.argmax(x, axis=1))
		print "%d out of %d correct" % (correct_cnt, output.shape[0])

if __name__ == '__main__':

	learning_rate = 0.001
	minibatch_size = 20
	valid_freq = 2 # validate one every 2 epoch

	#load data
	train, valid, test = cPickle.load(gzip.open('mnist.pkl.gz','rb'))
	t_x, t_y = train
	v_x, v_y = valid
	tt_x, tt_y = test

	n, n_in = t_x.shape
	n_out = 10  # number of classes
	assert n % minibatch_size == 0
	print "Model: %d features, %d output classes, %d training samples" % (n_in, n_out, n)

	train_x = numpy.append(t_x, numpy.ones(shape=(n,1)), 1)
	#turn y into a matrix of n_out * n with approximate class label column set
	train_y = numpy.zeros(shape=(n,n_out))
	train_y[numpy.arange(n),t_y] = 1 
	print "train_y.shape ", train_y.shape

	valid_x = numpy.append(v_x, numpy.ones(shape=(v_x.shape[0],1)), 1)


	#a two-layer NN
	nn = NeuralNet(units_array=[n_in,500, n_out])
	for i in range(100):
		for j in range(n/minibatch_size):
			ss = j*minibatch_size
			se = (j+1)*minibatch_size
			if (j % 500) == 0:
				print "j= %d out of %d" % (j, n/minibatch_size)
			nn.update(train_x[ss:se,:],train_y[ss:se,:],t_y[ss:se],learning_rate)

		if i % valid_freq == 0: 
			pright = nn.validate(valid_x, numpy.transpose(v_y))

		print "finish epoch ", i
	
