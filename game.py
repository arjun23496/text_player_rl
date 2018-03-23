__author__='Rakesh R Menon'
'''
File adapted from https://github.com/carpedm20/text-based-game-rl-tensorflow/blob/master/games/home.py
'''

from environment import Environment
from client import TCPClient
import os
import string
import time
import numpy as np
import re
import pdb

class MUD_Home_Environment(Environment):

	def __init__(self, args):
		'''
		Instantiating the MUD Home environment.
		'''

		super(MUD_Home_Environment, self).__init__()

		self.client = TCPClient()
		self.client.get()

		for key in args:
			setattr(self, key, args[key])

		self.name = "Home"
		self.rooms = ["Living", "Garden", "Kitchen", "Bedroom"]
		self.actions = ["eat", "sleep", "watch", "exercise", "go"]
		self.objects = ["north", "south", "east", "west"]
		self.quests = ["You are hungry", "You are sleepy", \
						"You are bored", "You are getting fat"]
		self.quests_mislead = ["You are not hungry", "You are not sleepy", \
								"You are not bored", "You are not getting fat"]
		self.quest_actions = ["eat", "sleep", "watch" ,"exercise"] # aligned to quests above
		self.idx_to_word = ["but", "now", "<pad>"]
		self.make_vocab(os.path.join(self.game_dir, "evennia/contrib/text_sims/build.ev"))

		self.login(self.username, self.password)

	def make_vocab(self, fname):
		'''
		Creating the vocabulary of the entire task.

		Inputs:

			fname : filename where the descriptions of all the objects and textual descriptions of the worlds are provided.

		'''
		with open(fname) as f:
			data = []
			for line in f:
				words = line.split()
				if words:
					if words[0] == '@detail' or words[0] == '@desc':
						self.idx_to_word.extend(words[3:])
					elif words[0] == '@create/drop':
						self.objects.append(words[1].split(":")[0])

		for quest in self.quests:
			quest = quest.translate(None, string.punctuation)
			self.idx_to_word.extend(quest.split())

		self.idx_to_word = list(set([word.lower().translate(None, string.punctuation) for word in self.idx_to_word]))

		self.idx_to_word.append("OOV")
		self.word_to_idx = {word: idx for idx, word in enumerate(self.idx_to_word)}

	def reset(self):
		'''
		Initiate a new game/restart in the environment.

		Outputs:

			state : (String) The initial state of the agent in the environment.
			reward : (Float) The reward received upon coming in the initial state.
			done : (Boolean) Is the episode over ?
		'''
		self.quest_checklist = []
		self.mislead_quest_checklist = []
		self.steps = 0
		self.random_teleport()
		self.random_quest()

		return self.get_state(timeout=True)



	def login(self, username, password):
		'''
		Method used for logging into the MUD Game environment.

		Inputs:

			username : (String) The username of the player.
			password : (String) The password of the player.
		'''

		self.client.send('connect {0} {1}'.format(username, password))
		text = self.client.get(3)
		if "You" not in text and "This" not in text:
			self.client.send('@batchcommand text_sims.build')
			self.client.get(67)
			self.client.send('start')

	def random_teleport(self):
		'''
		Method used for sending the agent to a random room in the environment.
		'''
		room_idx = np.random.randint(self.num_rooms)
		self.client.send('@tel tut#0%s' % room_idx)
		self.client.get()
		self.client.send('l')
		self.client.get()
		# print(" [*] Start Room : %s %s" % (room_idx, self.rooms[room_idx]))
	
	def random_quest(self):
		'''
		Method used for generating the quest order for the RL agent.
		'''
		idxs = np.random.permutation(len(self.quests))
		for idx in xrange(self.quest_levels):
			self.quest_checklist.append(idxs[idx])

		self.mislead_quest_checklist = [idxs[-1]]
		for idx in xrange(len(self.quest_checklist) - 1):
			self.mislead_quest_checklist.append(idxs[idx])

		# print(" [*] Start Quest : %s %s." % (self.get_quest_text(self.quest_checklist[0]), self.actions[self.quest_checklist[0]]))

	def get_state(self, action=None, object_=None, timeout=False):
		'''
		Method used for generating the current state of the RL agent in the environment.

		Inputs:

			action : (String) The action taken at the previous timestep.
			object : (String) The object chosen at the previous timestep.
			timeout : (Float)

		Outputs:

			state : (String) The current state of the RL agent in the environment.
			reward : (Float) The reward that the agent has received for performing the action in the previous timestep.
			is_finished : (Boolean) Is the episode over?
		'''
		is_finished = self.steps > self.max_step
		result = self.client.get(timeout=timeout)
		self.client.send('look')
		room_description = self.client.get()
		texts, reward = self.parse_game_output(result, room_description)
		# if self.debug:
		# log = " [@] get_state(\n\tdescription\t= %s \n\tquest\t\t= %s " % (texts[0], texts[1])
		# if action != None and object_ != None:
		# 	log += "\n\taction\t\t= %s %s " % (action, object_)
		# 	log += "\n\tresult\t\t= %s)" % (result)
		# log += "\n\treward\t\t= %s)" % (reward)
		# print(log)
		# if reward > 0:
		# 	time.sleep(2)

		# remove completed quest and refresh new quest
		if reward == 1:
			self.quest_checklist = self.quest_checklist[1:]
			self.mislead_quest_checklist = self.mislead_quest_checklist[1:]

			if len(self.quest_checklist) == 0:
				is_finished = True
			else:
				texts.append(self.get_quest_text(self.quest_checklist[0]))

		state = texts
		return state, reward, is_finished

	def parse_game_output(self, text, room_description):
		'''
		Method to generate the reward and the next state "string" description.

		Inputs:

			text : (String) Feedback from the environment after perfoming action.
			room_description : (String) Description of the room that the agent has moved into/is currently in.

		Outputs:

			text_to_agent : (String) The description of the current environmental state that goes to the agent.
			reward : (Float) Reward for last action.
		'''
		reward = None
		text_to_agent = [room_description, self.get_quest_text(self.quest_checklist[0])]

		if 'REWARD' in text:
			if self.quest_actions[self.quest_checklist[0]] in text:
				reward = int(re.search(r'\d+', text).group())

		if 'not available' in text or 'not find' in text:
			reward = self.junk_cmd_reward

		if reward == None:
			reward = self.default_reward

		return text_to_agent, reward

	def get_quest_text(self, quest_num):
		'''
		Method to generate the quest description.
		'''

		return self.quests_mislead[self.mislead_quest_checklist[0]] + " now but " + self.quests[quest_num] + " now."

	def step(self, action):
		'''
		Method to perform action in the environment.

		Inputs:

			action : (String) The action as suggested by the environment

		Outputs:

			state : (String) The current state of the RL agent in the environment.
			reward : (Float) The reward that the agent has received for performing the action in the previous timestep.
			is_finished : (Boolean) Is the episode over?
		'''
		self.client.send(action)
		action = action.split(" ")
		return self.get_state(action=action[0], object_=action[1])