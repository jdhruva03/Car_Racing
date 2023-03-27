import gym
from matplotlib import pyplot as plt
import numpy as np
import random
import keras
import cv2
# from replay_buffer import ReplayBuffer
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from collections import deque
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

DIMENSION = 96
DECAY_RATE = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_ITERS = 21500
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 1000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
TAU = 0.01
STRAIGHT = 20
NUM_FRAMES = 10
RGB = 3
EXPLORATION_PROB = 0.1

LARGE_CNN = True
MORE_ACTIONS = False


class DeepRL():
	def __init__(self, num_actions):
		self.num_actions = num_actions
		self.construct_convs()
		

	def construct_convs(self):

		if not LARGE_CNN:
			self.model = Sequential()
			self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(DIMENSION, DIMENSION*RGB, NUM_FRAMES)))
			self.model.add(Activation('relu'))
			self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
			self.model.add(Activation('relu'))
			self.model.add(Convolution2D(64, 3, 3))
			self.model.add(Activation('relu'))
			self.model.add(Flatten())
			self.model.add(Dense(512))
			self.model.add(Activation('relu'))
			self.model.add(Dense(self.num_actions))
			self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

			self.target_model = Sequential()
			self.target_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(DIMENSION, DIMENSION*RGB, NUM_FRAMES)))
			self.target_model.add(Activation('relu'))
			self.target_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
			self.target_model.add(Activation('relu'))
			self.target_model.add(Convolution2D(64, 3, 3))
			self.target_model.add(Activation('relu'))
			self.target_model.add(Flatten())
			self.target_model.add(Dense(512))
			self.target_model.add(Activation('relu'))
			self.target_model.add(Dense(self.num_actions))
			self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
			self.target_model.set_weights(self.model.get_weights())
		else:
			self.model = Sequential()
			self.model.add(Convolution2D(64, 8, 8, subsample=(4, 4), input_shape=(DIMENSION, DIMENSION*RGB, NUM_FRAMES)))
			self.model.add(Activation('relu'))
			self.model.add(Convolution2D(128, 4, 4, subsample=(2, 2)))
			self.model.add(Activation('relu'))
			self.model.add(Convolution2D(128, 3, 3))
			self.model.add(Activation('relu'))
			self.model.add(Flatten())
			self.model.add(Dense(1024))
			self.model.add(Activation('relu'))
			self.model.add(Dense(self.num_actions))
			self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

			self.target_model = Sequential()
			self.target_model.add(Convolution2D(64, 8, 8, subsample=(4, 4), input_shape=(DIMENSION, DIMENSION*RGB, NUM_FRAMES)))
			self.target_model.add(Activation('relu'))
			self.target_model.add(Convolution2D(128, 4, 4, subsample=(2, 2)))
			self.target_model.add(Activation('relu'))
			self.target_model.add(Convolution2D(128, 3, 3))
			self.target_model.add(Activation('relu'))
			self.target_model.add(Flatten())
			self.target_model.add(Dense(1024))
			self.target_model.add(Activation('relu'))
			self.target_model.add(Dense(self.num_actions))
			self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
			self.target_model.set_weights(self.model.get_weights())

	def predict_movement(self, data, epsilon):
		"""Predict movement of game controler where is epsilon
		probability randomly move."""
		q_actions = self.model.predict(data.reshape(1, DIMENSION, DIMENSION*RGB, NUM_FRAMES), batch_size = 1)
		opt_policy = np.argmax(q_actions)
		rand_val = np.random.random()
		if rand_val < epsilon:
			opt_policy = np.random.randint(0, self.num_actions)
		return opt_policy, q_actions[0, opt_policy]

	def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
		"""Trains network to fit given parameters"""
		batch_size = s_batch.shape[0]
		targets = np.zeros((batch_size, self.num_actions))

		for i in range(batch_size):
			# print(self.model.predict(s_batch[i].reshape(1, DIMENSION, DIMENSION, 3), batch_size = 1))
			targets[i] = self.model.predict(s_batch[i].reshape(1, DIMENSION, DIMENSION*RGB, NUM_FRAMES), batch_size = 1)
			fut_action = self.target_model.predict(s2_batch[i].reshape(1, DIMENSION, DIMENSION*RGB, NUM_FRAMES), batch_size = 1)
			targets[i, a_batch[i]] = r_batch[i]
			if d_batch[i] == False:
				targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

		s_batch.resize(batch_size, DIMENSION, DIMENSION*RGB, NUM_FRAMES)
		loss = self.model.train_on_batch(s_batch, targets)

	def target_train(self):
		model_weights = self.model.get_weights()
		target_model_weights = self.target_model.get_weights()
		for i in range(len(model_weights)):
			target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
		self.target_model.set_weights(target_model_weights)

	def save_network(self, path):
		# Saves model at specified path as h5 file
		self.model.save(path)
		print("Successfully saved network.")

	def load_network(self, path):
		self.model = load_model(path)
		print("Succesfully loaded network.")


class CarRacer():
	def __init__(self, file_name, load=False):
		self.file_name = file_name
		self.env = gym.make('CarRacing-v0')
		self.env.reset()
		
		steer = np.array([-0.25, 0.0, 0.25])
		gas = np.array([0.1, 0.4, 0.8])
		brake = np.array([0, 0.1, 0.25])
		if MORE_ACTIONS:
			steer = np.linspace(-1, 1, 5)
			gas = np.linspace(0.2, 1, 5)
			brake = np.linspace(0, 0.2, 5)
		self.action_space = np.array([[i, j, k] for i in steer for j in gas for k in brake])
		self.deep_rl = DeepRL(len(self.action_space))
		if load:
			self.deep_rl.load_network(file_name + ".h5")

		self.state_buffer = []
		for _ in range(NUM_FRAMES):
			s = self.env.step(self.action_space[12])[0]
			self.state_buffer.append(s)

		self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

	def plot_q_vals(self):
		with open(self.file_name + ".txt", "r") as f:
			line = f.readline()
			q_vals = [float(i) for i in line.split()]
			# q_vals = q_vals[0:94]
			iters = np.linspace(0, (len(q_vals)-1)*100, len(q_vals))
			plt.plot(iters, q_vals)
			plt.title('Predicted Q value vs. iteration number: ' + self.file_name)
			plt.ylabel('predicted Q value')
			plt.xlabel('iteration')
			plt.show()

	def convert_state(self):
		state = np.array(self.state_buffer)
		state = np.reshape(state, (DIMENSION, -1, NUM_FRAMES))
		return state

	def run(self, num_games):
		games_so_far = 0
		epsilon = 0
		alive_frame = 0
		total_reward = 0
		rewards = np.zeros(num_games)
		while games_so_far < num_games:
			initial_state = self.convert_state()
			self.state_buffer = []
			predict_action_idx, predict_q_value = self.deep_rl.predict_movement(initial_state, epsilon)
			predict_action = self.action_space[predict_action_idx]

			reward = 0
			done = False
			for i in range(NUM_FRAMES):
				obs, temp_reward, temp_done, _ = self.env.step(predict_action)
				self.env.render()
				reward += temp_reward
				self.state_buffer.append(obs)
				done = done | temp_done

			total_reward += reward

			if done:
				print("Lived with maximum time ", alive_frame)
				print("Earned a total of reward equal to ", total_reward)
				self.env.reset()
				rewards[games_so_far] = total_reward
				games_so_far += 1
				alive_frame = 0
				total_reward = 0
		return rewards


	def train(self, num_iters):
		observation_num = 0
		# curr_state = self.convert_state()
		epsilon = INITIAL_EPSILON
		alive_frame = 0
		total_reward = 0

		while observation_num < num_iters:
			# self.env.render()
			if observation_num % 1000 == 999:
				print(("Executing loop %d" %observation_num))

			# Slowly decay the learning rate
			if epsilon > FINAL_EPSILON:
				epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

			initial_state = self.convert_state()
			self.state_buffer = []

			# if observation_num < STRAIGHT:
			# 	predict_action_idx, predict_q_value = 4, 0
			# else:

			# if np.random.uniform() <= EXPLORATION_PROB:
			# 	predict_action_idx, predict_q_value = np.random.choice(range(NUM_FRAMES)), 0
			# else:
			predict_action_idx, predict_q_value = self.deep_rl.predict_movement(initial_state, epsilon)

			predict_action = self.action_space[predict_action_idx]

			reward = 0
			done = False
			for i in range(NUM_FRAMES):
				obs, temp_reward, temp_done, _ = self.env.step(predict_action)
				reward += temp_reward
				self.state_buffer.append(obs)
				done = done | temp_done

			if observation_num % 100 == 0:
				print("We predicted a q value of ", predict_q_value, "at obs num ", observation_num)
				f = open(self.file_name + ".txt", "a+")
				f.write(str(predict_q_value) + " ")
				f.close()

			if done:
				print("Lived with maximum time ", alive_frame)
				print("Earned a total of reward equal to ", total_reward)
				self.env.reset()
				alive_frame = 0
				total_reward = 0

			new_state = self.convert_state()

			self.replay_buffer.add(initial_state, predict_action_idx, reward, done, new_state)
			total_reward += reward

			if self.replay_buffer.size() > MIN_OBSERVATION:
				s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(MINIBATCH_SIZE)
				self.deep_rl.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
				self.deep_rl.target_train()

			# Save the network every 1000 iterations
			if observation_num % 1000 == 999:
				print("Saving Network")
				self.deep_rl.save_network(self.file_name + ".h5")

			alive_frame += 1
			observation_num += 1
		self.env.close()



class ReplayBuffer():
	"""Constructs a buffer object that stores the past moves
	and samples a set of subsamples"""
	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.count = 0
		self.buffer = deque()

	def add(self, s, a, r, d, s2):
		"""Add an experience to the buffer"""
		# S represents current state, a is action,
		# r is reward, d is whether it is the end, 
		# and s2 is next state
		experience = (s, a, r, d, s2)
		if self.count < self.buffer_size:
			self.buffer.append(experience)
			self.count += 1
		else:
			self.buffer.popleft()
			self.buffer.append(experience)

	def size(self):
		return self.count

	def sample(self, batch_size):
		"""Samples a total of elements equal to batch_size from buffer
		if buffer contains enough elements. Otherwise return all elements"""
		batch = []
		if self.count < batch_size:
			batch = random.sample(self.buffer, self.count)
		else:
			batch = random.sample(self.buffer, batch_size)

		# Maps each experience in batch in batches of states, actions, rewards
		# and new states
		s_batch, a_batch, r_batch, d_batch, s2_batch = list(map(np.array, list(zip(*batch))))

		return s_batch, a_batch, r_batch, d_batch, s2_batch

	def clear(self):
		self.buffer.clear()
		self.count = 0


if __name__ == "__main__":
	assert len(sys.argv) == 2
	file_name = sys.argv[1]
	car_racer = CarRacer(file_name, load=True)
	car_racer.train(TOT_ITERS)
	car_racer.plot_q_vals()
	rew = car_racer.run(3)
	print(float(sum(rew))/len(rew))