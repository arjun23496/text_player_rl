__author__='Rakesh R Menon'

import numpy as np
import torch
import torchvision
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from model import Model
from replay_buffer import ReplayBuffer
import pdb
import pickle
import os

class LSTMDQN(nn.Module, Model):

	'''
	Creating the LSTM DQN model from Karthik's paper.
	'''

	def __init__(self, args):
		'''
		Instantiating the object and setting all parameters as object attirbutes. Also building model.
		'''

		super(LSTMDQN, self).__init__()

		for key in args:
			setattr(self, key, args[key])

		self.build_model()

	def build_model(self):
		'''
		Method to be used for constructing the model architecture.
		'''
		#Create Word Embedding in Pytorch
		self.network_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
		#Network Architecture
		self.network_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
		self.network_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.network_linear_action = nn.Linear(self.hidden_dim, self.num_actions, bias=False)
		self.network_linear_object = nn.Linear(self.hidden_dim, self.num_objects, bias=False)
		self.network_hidden = self._init_hidden(self.minibatch_size)

	def _init_hidden(self, minibatch_size):
		'''
		Method to set the initial hidden state of the lstm to zeros.

		Inputs:

			minibatch_size : (Int) Size of the minibatch that is about to be input to the LSTM

		Outputs:

			hidden : (Variable) Initialisations for the hidden state and cell states of the LSTM
		'''
		hidden = (autograd.Variable(torch.zeros(1, minibatch_size, self.hidden_dim)),
			autograd.Variable(torch.zeros((1, minibatch_size, self.hidden_dim))))
		return hidden

	def forward(self, inputs, seq_lengths, minibatch_size):
		'''
		Method used for performing forward pass of data through the network.


		Inputs : 

			inputs:  (Long Tensor Variable) word-to-index versions of the sentence(s)
			seq_lengths : (List) Length of the sentences
			minibatch_size : (Int) Length of minibatch

		Outputs:
	
			out_action : (Float Tensor) Action Q-values
			out_object : (Float Tensor) Ojbect Q-values
		'''
		self.network_hidden = self._init_hidden(minibatch_size)
		embeds = self.network_embedding(inputs)
		embeds = embeds.transpose(0,1)
		embeds = pack_padded_sequence(embeds, np.array(seq_lengths))
		out_scores, self.network_hidden = self.network_lstm(embeds, self.network_hidden)
		out_scores, _ = pad_packed_sequence(out_scores)
		
		# out_scores = torch.sum(out_scores, dim=0)
		# seq_lengths = autograd.Variable(torch.Tensor(seq_lengths))
		# seq_lengths.require_grad = False
		# out_vals = torch.div(out_scores, seq_lengths.view(-1, 1))
		out_vals = autograd.Variable(torch.zeros(out_scores.shape[1],out_scores.shape[2]))
		for i, val in enumerate(seq_lengths):
			out_vals[i] = torch.mean(out_scores[:min(val, 30)][:,i], dim=0)	

		out_vals = F.relu(out_vals)
		out_vals = self.network_linear(out_vals)
		out_vals = F.relu(out_vals)

		#Action output
		out_action = self.network_linear_action(out_vals)

		#Object output
		out_object = self.network_linear_object(out_vals)

		return out_action, out_object

	# def __getstate__(self):
	# 	'''
	# 	Method used for defining all the model param values that will be saved in pickle file.

	# 	Output:

	# 		state : (Dictionary) contains the weights of all the network components.
	# 	'''
		
	# 	# if not os.path.isdir('./models'):
	# 	# 	os.makedirs('./models/')

	# 	# torch.save(self.state_dict(), './models/lstmdqn.model')


	# 	# return self.__dict__
	# 	pass
		


	# def __setstate__(self, state):
	# 	'''
	# 	Method used for loading all the variables from a pickle file.
	# 	'''

	# 	# if not os.path.isdir('./models'):
	# 	# 	raise ValueError('No previous model file to load.')

	# 	# self.load_state_dict(torch.load('./models/lstmdqn.model'))

	# 	# for key in state.keys():
	# 	# 	setattr(self, key, state[key])
	# 	pass
