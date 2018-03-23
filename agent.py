__author__='Rakesh R Menon'


import numpy as np
import random
import torch
import torchvision
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from lstmdqn import LSTMDQN
from replay_buffer import ReplayBuffer
import os
import pdb
import pickle
import string


class Agent:

	def __init__(self, args):
		'''
		Method to instantiate the model.
		'''

		for key in args:
			setattr(self, key, args[key])

		self.replay_buffer = ReplayBuffer(self.max_buffer_size)

		args['vocab_size'] = len(self.idx_to_word)
		
		self.model = LSTMDQN(args)
		self.model_target = LSTMDQN(args)

		# self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

		self.update_target()

	def update_target(self):
		'''
		Method to update the target network of the agent.
		'''
		self.model_target.load_state_dict(self.model.state_dict())

		for parameter in self.model_target.parameters():
			parameter.requires_grad = False

	def sentence_to_idx(self, sentence, word_to_idx):
		'''
		Method to convert sentence to word-to-idx format.

		Inputs:

			sentence : (List) A sentence
			word_to_idx : (Dict) A dictionary consisting of the word-to-idx conversions.

		Outputs:

			(List) word-to-idx sentences.
		'''
		return [word_to_idx.get(word.lower(),word_to_idx['OOV']) for word in sentence]

	def state_preprocessing(self, sentences):
		'''
		Method to pre-process the sentences to word-to-idx form.

		Input :
			
			input : (List) List of sentences.

		Outputs:

			sentence_lstm : (List) List of word-to-idx sentences (in descending order of length of sentences)
			seq_length : (List) List of lengths of sentences (in descending order of length of sentences)
		'''
		
		sentences = [(" ").join(sentence).translate(None, string.punctuation) for sentence in sentences]
		
		sentence_batch = [sentence.split(" ") for sentence in sentences]
		
		seq_length = map(len,sentence_batch)
		indices = np.argsort(seq_length)
		seq_length = list(np.array(seq_length)[indices])
		max_length = np.amax(seq_length)
		sentence_batch = [sentence_batch[i] for i in indices]
		sentence_batch = sentence_batch[::-1]
		seq_length = seq_length[::-1]

		sentence_lstm = []
		for sentence in sentence_batch:
			senten = sentence[:] + ['<pad>']*(max_length-len(sentence))
			sentence_idx = self.sentence_to_idx(senten, self.word_to_idx)
			sentence_lstm.append(sentence_idx)


		return sentence_lstm, seq_length



	def predict(self, input, minibatch_size, model):
		'''
		Method to predict the action of the reinforcement learning agent.

		Inputs:

			input : (List) List of sentences.

		Outputs:

			out_action : (Numpy float array) Q-values of action.
			out_object : (Numpy float array) Q-values of object.
		'''

		input_batch, seq_length = self.state_preprocessing(input)
		out_action, out_object = model.forward(autograd.Variable(torch.LongTensor(input_batch)), seq_length, minibatch_size)

		return out_action, out_object

	def update(self):
		'''
		Method to perform Q-learning update on a minibatch of samples stored in the replay memory.
		'''

		input_sentences, action, rewards, next_sentences, done = self.replay_buffer.sample(self.minibatch_size)

		action = [act_.split(" ") for act_ in action]

		action_word = np.array(action)[:, 0].tolist()
		object_word = np.array(action)[:, 1].tolist()

		states_action = []
		for act in action_word:
			for i in range(len(self.actions)):
				if self.actions[i]==act:
					states_action.append(i)

		states_object = []
		for obj in object_word:
			for i in range(len(self.objects)):
				if self.objects[i]==obj:
					states_object.append(i)

		minibatch_size = len(action)
		
		next_action, next_object = self.predict(next_sentences, minibatch_size, self.model_target)

		self.model.zero_grad()
		input_action, input_object = self.predict(input_sentences, minibatch_size, self.model)

		Q_action = input_action[np.arange(len(states_action)), states_action]
		Q_object = input_object[np.arange(len(states_object)), states_object]

		Q_prime_action, _ = torch.max(next_action, dim=1)
		Q_prime_object, _ = torch.max(next_object, dim=1)

		rewards = Variable(torch.Tensor(rewards))
		rewards.requires_grad = False

		done = Variable(torch.Tensor([int(not don) for don in done]))
		done.requires_grad = False

		target_q_action = self.gamma*torch.mul(Q_prime_action, done)
		target_q_object = self.gamma*torch.mul(Q_prime_object, done)
		target_q = (target_q_action + target_q_object)/2

		loss_object = (rewards+target_q - Q_object)
		loss_action = (rewards+target_q - Q_action)

		loss_action = torch.clamp(loss_action, -self.clip_delta, self.clip_delta)
		loss_object = torch.clamp(loss_object, -self.clip_delta, self.clip_delta)

		loss_action = (0.5*loss_action**2).sum()
		loss_object = (0.5*loss_object**2).sum()

		loss_action.backward(retain_graph=True)
		loss_object.backward(retain_graph=True)

		self.optimizer.step()

	def pick_action(self, input):
		'''
		Method to pick an action given the current state using epsilon-greedy action selection.

		Inputs:

			input : (List) Sentence.

		Outputs:

			action : (String) The action to be taken by the agent. (in words)

		'''


		if random.random() <= self.epsilon:
			action_index = random.randrange(0, self.num_actions)
			object_index = random.randrange(0, self.num_objects)
		else:
			scores_action, scores_object = self.predict([input], 1, self.model)
			action_softmax = F.softmax(scores_action, dim=1).data.numpy()
			object_softmax = F.softmax(scores_object, dim=1).data.numpy()
			action_index = np.random.choice(np.where(action_softmax[0]==np.amax(action_softmax))[0])
			object_index = np.random.choice(np.where(object_softmax[0]==np.amax(object_softmax))[0])

		action_word = self.actions[action_index]
		object_word = self.objects[object_index]

		action = " ".join([action_word, object_word])

		return action


	def pick_action_test(self, input):
		'''
		Method to pick an action given the current state using greedy action selection.

		Inputs:

			input : (List) Sentence.

		Outputs:

			action : (String) The action to be taken by the agent. (in words)

		'''

		scores_action, scores_object = self.predict([input], 1, self.model)

		action_softmax = F.softmax(scores_action, dim=1).data.numpy()
		object_softmax = F.softmax(scores_object, dim=1).data.numpy()

		if random.random() <= 0.05:
			action_index = random.randrange(0, self.num_actions)
			object_index = random.randrange(0, self.num_objects)
		else:
			action_index = np.random.choice(np.where(action_softmax[0]==np.amax(action_softmax))[0])
			object_index = np.random.choice(np.where(object_softmax[0]==np.amax(object_softmax))[0])

		action_word = self.actions[action_index]
		object_word = self.objects[object_index]

		action = " ".join([action_word, object_word])

		return action

	def store(self, transition_tuple):
		'''
		Storing the s_t, a_t, r_t, s_{t+1} tuple in the replay memory
		'''
		self.replay_buffer.store(transition_tuple)