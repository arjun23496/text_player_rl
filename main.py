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
import argparse
from agent import Agent
from game import MUD_Home_Environment
import matplotlib.pyplot as plt
import pickle
import pdb


parser = argparse.ArgumentParser(description='LSTMDQN Home World')
parser.add_argument('--learning-rate', type=float, default= 0.0005, help='learning rate (default:  0.0005)')
parser.add_argument('--gamma', type=float, default=0.5, help='discount factor for rewards (default: 0.5)')
parser.add_argument('--start-epsilon', type=float, default=1.00, help='initial epsilon for exploration (default: 1.00)')
parser.add_argument('--final-epsilon', type=float, default=0.20, help='final epsilon for exploration (default: 0.20)')
parser.add_argument('--epsilon-decay-time', type=int, default=100000, help='# of timesteps after which to decay epsilon to final value (default: 100000)')
parser.add_argument('--embedding-dim', type=int, default=20, help='word embedding dimensions (default: 20)')
parser.add_argument('--hidden-dim', type=int, default=20, help='hidden state dimensions (default: 20)')
parser.add_argument('--max-step', type=int, default=20, help='maximum length of an episode (default: 20)')
parser.add_argument('--game-dir', type=str, default="/Users/rakeshrmenon/COMPSCI690NProject/text-world", help='the name of game directory home')
parser.add_argument('--model-save', type=str, default="./models/", help='directory to save the model in (default: ../models/)')
parser.add_argument('--model-save-freq', type=int, default=5, help='model saving frequency (in epochs) (default: 5)')
parser.add_argument('--update-target-freq', type=int, default=1000, help='update target network frequency (default: 1000)')
parser.add_argument('--start-learn', type=int, default=1000, help='start learning (default: 1000)')
parser.add_argument('--num-rooms', type=int, default=4, help='number of rooms in the home environment (default: 4)')
parser.add_argument('--default-reward', type=float, default=-0.01, help='default reward to be given at each timestep (default: -0.01)')
parser.add_argument('--junk-cmd-reward', type=float, default=-0.1, help='penalty to be given for invalid action (default: -0.10)')
parser.add_argument('--quest-levels', type=int, default=1, help='number of quests to complete (default: 1)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to complete (default: 100)')
parser.add_argument('--minibatch-size', type=int, default=64, help='minibatch size for training (default: 64)')
parser.add_argument('--username', type=str, default="root", help='the username of the agent (default: root)')
parser.add_argument('--password', type=str, default="root", help='the password for the agent (default: root)')
parser.add_argument('--train', type=bool, default=True, help='is training? (default: True)')
parser.add_argument('--train-steps', type=int, default=1000, help='number of training steps (default: 1000)')
parser.add_argument('--eval-steps', type=int, default=1000, help='number of evaluation steps (default: 1000)')
parser.add_argument('--update-gradient', type=int, default=4, help='frequency of gradient update (in steps) (default: 4)')
parser.add_argument('--max-buffer-size', type=int, default=100000, help='replay memory size (default: 100000)')
parser.add_argument('--clip-delta', type=int, default=1, help='td-error clipping (default: 1)')

def main():

	args = parser.parse_args()

	epochs = 0
	env = MUD_Home_Environment(vars(args))
	setattr(args, 'actions', env.actions)
	setattr(args, 'objects', env.objects)
	setattr(args, 'word_to_idx', env.word_to_idx)
	setattr(args, 'idx_to_word', env.idx_to_word)
	setattr(args, 'num_actions', len(env.actions))
	setattr(args, 'num_objects', len(env.objects))
	setattr(args, 'epsilon', args.start_epsilon)
	lstm_agent = Agent(vars(args))
	reward_over_train_epochs = []
	reward_over_test_epochs = []
	for epoch in range(args.epochs):

		print "Epoch : {}".format(epoch+1)

		#Training
		quest_completion = 0
		episode_count = 0
		reward_over_episodes = []
		overall_steps = 0
		while (overall_steps< args.train_steps):

			steps = 0
			rewards = 0.0
			state, reward, done = env.reset()
			while (steps < args.max_step) and (not done):

				lstm_agent.epsilon = (args.start_epsilon- args.final_epsilon) * (args.epsilon_decay_time - max(0, epoch*args.train_steps + overall_steps + steps - args.start_learn)) / args.epsilon_decay_time + args.final_epsilon

				action = lstm_agent.pick_action(state)
				next_state, reward, done = env.step(action)

				# if epoch==0:
				# 	print state, action, reward

				if len(lstm_agent.replay_buffer)==args.max_buffer_size : lstm_agent.replay_buffer.popleft()
				lstm_agent.store((state, action, reward, next_state, done))

				state = next_state
				rewards += reward
				steps +=1

				if (epoch*args.train_steps + overall_steps + steps + 1)>=args.start_learn and (overall_steps + steps)%args.update_gradient==0:  lstm_agent.update()

				if (overall_steps + steps + 1)%args.update_target_freq==0: lstm_agent.update_target()

			reward_over_episodes.append(np.around(rewards, decimals=2))

			if done : quest_completion += 1
			episode_count +=1

			overall_steps += steps

		reward_over_train_epochs.append(np.mean(reward_over_episodes))
		print "Training Rewards : {}".format(np.mean(reward_over_episodes))
		# print "Training Quest Completion : {}".format(float(quest_completion)/episode_count)

		#Testing
		quest_completion = 0
		episode_count = 0
		reward_over_episodes = []
		overall_steps = 0
		while (overall_steps<=args.eval_steps):

			steps = 0
			rewards = 0.0
			state, reward, done = env.reset()
			while (steps < args.max_step) and (not done):

				action = lstm_agent.pick_action_test(state)
				next_state, reward, done = env.step(action)

				state = next_state
				rewards += reward
				steps +=1

			if done : quest_completion += 1
			episode_count +=1

			reward_over_episodes.append(np.around(rewards, decimals=2))

			overall_steps += steps

		reward_over_test_epochs.append(np.mean(reward_over_episodes))
		print "Testing Rewards : {}".format(np.mean(reward_over_episodes))
		# print "Testing Quest Completion : {}".format(float(quest_completion)/episode_count)

		print "Saving model..."
		pickle.dump(lstm_agent, open("./models/lstmdqn.params", "wb"))


	plt.figure()
	plt.plot(range(args.epochs), reward_over_train_epochs, label='Train')
	plt.plot(range(args.epochs), reward_over_test_epochs, label='Test')
	plt.legend()
	plt.xlabel('Number of epochs')
	plt.ylabel('Score for episode')
	plt.title('Rewards obtained over epochs during training and testing')
	filename = 'HomeWorld_rewards.png'
	plt.savefig(filename, dpi=300)


if __name__=='__main__':
	main()