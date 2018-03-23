__author__='Rakesh R Menon'

import random
from collections import deque

class ReplayBuffer:

	def __init__(self, max_buffer_size):

		'''
		Instantiating the replay buffer for the agent
		'''

		self.buffer_size = max_buffer_size
		self.replay_buffer = deque(maxlen=max_buffer_size)

	def store(self, transition_tuple):
		'''
		Storing the s_t, a_t, r_t, s_{t+1} tuple in the replay memory
		'''
		self.replay_buffer.append(transition_tuple)

	def sample(self, batch_size):
		'''
		Sampling from the replay memory
		'''
		batch_size = min(batch_size, len(self.replay_buffer))
		samples = random.sample(self.replay_buffer, batch_size)

		return zip(*samples)

	def popleft(self):
		'''
		Remove element from the beginning of deque.
		'''
		self.replay_buffer.popleft()

	def clear_buffer(self):
		'''
		Clear replay memory.
		'''
		self.replay_buffer.clear()

	def __len__(self):
		return len(self.replay_buffer)