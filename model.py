__author__='Rakesh R Menon'


import abc


class Model:

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def build_model(self):
		'''
		Method to be used for constructing the model architecture.
		'''
		pass

	@abc.abstractmethod
	def forward(self):
		'''
		Method used for performing forward pass of data through the network/
		'''
		pass

	# @abc.abstractmethod
	# def __getstate__(self, filename):
	# 	'''
	# 	Method used for defining all the model param values that will be saved in pickle file.
	# 	'''
	# 	pass

	# @abc.abstractmethod
	# def __setstate__(self, filename):
	# 	'''
	# 	Method used for loading all the variables from a pickle file
	# 	'''
	# 	pass