__author__='Rakesh R Menon'

import abc

class Environment:

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def step(self, action):
		'''
		Method to incorporate action step for RL agent.
		'''
		pass

	@abc.abstractmethod
	def reset(self):
		'''
		Method to reset the environment and send the agent back to the start state
		'''
		pass