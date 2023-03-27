'''
Author: Jainam Dhruva
Title: Deep Q learning on autonomous car agent
'''


# python C:\Users\Jainam\anaconda3\envs\ml\Lib\site-packages\gymnasium\envs\box2d\car_racing.py

import gymnasium as gym
import torch 
import torch.nn
from collections import deque
import numpy as np
import random
from processimage import processimage


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# HYPER PARAMETERS
MIN_EPSILON = 0.05
MAX_EPSILON = 1
DECAY = 0.01
TOTAL_EPISODESS = 300
MAX_EXPERIENCES = 15000
TRAIN_TARGET_MODEL_STEPS = 100
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.618
BATCH_SIZE = 64
KEEP_PROB = 0.8


# Implementation of CNN/ConvNet Model
class DQN(torch.nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        # Conv -> (?, 28, 28, 32)
        # Pool -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=8, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - KEEP_PROB)
        )
        # L2 ImgIn shape=(?, 14, 14, 32)
        # Conv      ->(?, 14, 14, 64)
        # Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - KEEP_PROB)
        )
        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 7, 7, 128)
        # Pool ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - KEEP_PROB)
        )

        # L4 FC 506944 121 7744 inputs -> 64 outputs
        self.fc1 = torch.nn.Linear(7744, 64, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - KEEP_PROB)
        )
        # L5 Final FC 64 inputs -> 5 outputs
        self.fc2 = torch.nn.Linear(64, 5, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters

    def forward(self, x):
        # print(f"A Dimension of input: {x.shape}")
        out = self.layer1(x)
        # print(f"B Dimension of input after layer 1 (Conv2d): {out.shape}")
        out = self.layer2(out)
        # print(f"C Dimension of input after layer 2 (Conv2d): {out.shape}")
        out = self.layer3(out)
        # print(f"D Dimension of input after layer 3 (Conv2d): {out.shape}")
        out = out.view(out.size(0), -1)   # Flatten them for FC
        # print(f"E Dimension of input after Flattening (Conv2d): {out.shape}")
        # input("here2")
        out = self.fc1(out)
        # print(f"F Dimension of input after layer 4 (forward1): {out.shape}")
        # input("here3")
        out = self.fc2(out)
        # print(f"G Dimension of input after layer 5 (forward2): {out.shape}")
        # input("here4, by return")
        # print(out)
        return out


def train(env, replay_buffer, prediction_model, target_model, episode_done):
    # print("Training: ")
    torch.autograd.set_detect_anomaly(True)

    ## Setting the model to training model
    prediction_model.train()   

    ## HYPER PARAMETERS
    global LEARNING_RATE 
    global DISCOUNT_FACTOR 

    MIN_REPLAY_SIZE = 100
    ## Only train if atleast minimum replay size has been met
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return

    ## Create a batch for training
    global BATCH_SIZE
    mini_batch = random.sample(replay_buffer, BATCH_SIZE)
    
    ## Pass the current states through the prediction network. 
    ## Store the Q-values predicted with the prediction network.
    current_states = np.array([transition[0] for transition in mini_batch])
    # print("A current_states size = " + str(current_states.shape))
    current_states = np.expand_dims(current_states, axis=1)
    # print("B current_states size = " + str(current_states.shape))
    # print(current_states.dtype)
    current_states_tensor = torch.tensor(current_states)
    current_states_tensor = current_states_tensor.to(torch.float32)
    # print(f"Datatype of tensor: {current_states_tensor.dtype}")
    # print(f"Dimension of tensor: {current_states_tensor.shape}")
    
    current_qs_list = torch.clone(prediction_model(current_states_tensor))
    

    ## Pass the new states through the target network. 
    ## Store the Q-values predicted with the target network.
    new_states = np.array([transition[3] for transition in mini_batch])
    # print("A new_states size = " + str(new_states.shape))
    new_states = np.expand_dims(new_states, axis=1)
    # print("B new_states size = " + str(new_states.shape))
    
    new_states_tensor = torch.tensor(new_states)
    new_states_tensor = new_states_tensor.to(torch.float32)
    # print(f"Datatype of tensor: {new_states_tensor.dtype}")
    # print(f"Dimension of tensor: {new_states_tensor.shape}")
    
    future_qs_list = torch.clone(target_model(current_states_tensor))

    # print("Dimension of current Qs " + str(current_qs_list.shape))
    # print("Dimension of future Qs " + str(future_qs_list.shape))

    # input("here new")

    # X = []
    # Y = []
    
    # HYPER PARAMETERS
    CRITERION = torch.nn.MSELoss()
    OPTIMIZER = torch.optim.Adam(prediction_model.parameters(), lr=LEARNING_RATE)
    
   

    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        
        # Zero out the gradients
        OPTIMIZER.zero_grad()
        prediction_model.zero_grad()
        target_model.zero_grad()

        # Get the maximum of the future Q value
        if not done:
            max_future_q = reward + DISCOUNT_FACTOR * (torch.max(future_qs_list[index]))
        else:
            max_future_q = reward
            max_future_q = torch.tensor(max_future_q)

        current_qs = torch.clone(current_qs_list[index])
        current_q = current_qs[action]
        
        # print("todaloo")
        # print(type(max_future_q), type(current_q))
        # input("todalooolorl")

        loss = CRITERION(max_future_q.detach(), current_q)
        # loss.requires_grad = True
        # print('index ', index)
        loss.backward()

        OPTIMIZER.step()

        
        # current_qs[action] = (1 - LEARNING_RATE) * current_qs[action] + LEARNING_RATE * max_future_q

        # X.append(observation)
        # Y.append(current_qs)
    
    # model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def main():

    # Intitialize the Environment
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")
    
    # 1. Initialize the Target and Main models
    # Main Model (updated every X steps)
    prediction_model = DQN()
    # Target Model (updated every Y steps)
    target_model = DQN()

    target_model.load_state_dict(prediction_model.state_dict())

    print(prediction_model)
    print(target_model)

    # Initialize the Replay Buffer
    replay_buffer = deque(maxlen=MAX_EXPERIENCES)

    # Step Count to keep track of when to update prediction_model and target_model
    step_count = 0
    
    for episode in range(TOTAL_EPISODESS):

        total_training_rewards = 0
        
        current_state, info = env.reset()
        current_state = processimage.process_image(current_state)
        
        episode_done = False
        epsilon = 0.75

        print("EPISODE " + str(episode))

        while (not episode_done):

            step_count += 1 
            if(step_count%500==0):
                print("Step Count = " + str(step_count))

            # Use Q value to determine action
            random_number = np.random.random()
            if(random_number < epsilon):
                # Do random action
                action = env.action_space.sample()
                # print("Acttion random", action, type(action))  
                # TBD - assign more randomeness to accelerate so the car moves ahead

            else:
                # print("epsilon here")
                current_state_np = np.expand_dims(current_state, axis=0)
                current_state_np = np.expand_dims(current_state_np, axis=0)
                # print("current state shape ", current_state.shape)
                # input("yoo")
                current_state_tensor = torch.tensor(current_state_np)
                current_state_tensor = current_state_tensor.to(torch.float32)
                action_q_values = prediction_model(current_state_tensor)
                # print(action_q_values.shape)
                action = torch.argmax(action_q_values)
                action = np.int64(action)

                # print("Acttion predicted", action, type(action))  
                

            # 'current_  state/new_state' is an 96*96*3 array containing RGB values for the 96*96 pixel image
            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = processimage.process_image(new_state)
            
            if (terminated or truncated):
                print("Episode done omg")
                episode_done = True
            
            # Add the experience to the replay Buffer
            replay_buffer.append([current_state, action, reward, new_state, episode_done])

            #  Update the Main Network using the Bellman Equation
            if ( (step_count % 4 == 0) or episode_done):
                train(env, replay_buffer, prediction_model, target_model, episode_done)
                # dummy_function(step_count)

            current_state = new_state
            total_training_rewards += reward

            
            # In case the episode is done, handle that
            if (episode_done):
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1

                # If step count is greater than equal to train_model_steps
                if (step_count >= TRAIN_TARGET_MODEL_STEPS):
                    print('Copying main network weights to the target network weights')

                    # Copy the weights from prediction model to target model
                    with torch.no_grad():
                        for target_params, prediction_params in zip(target_model.parameters(), prediction_model.parameters()):
                            target_params = torch.clone(prediction_params)

                    step_count = 0
                break
            
        # Update epsilon after each episode
        # epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY * episode)

    
    
    env.close()

def dummy_main():
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")
    observation, info = env.reset()

    prediction_model = DQN()
    target_model = DQN()
    target_model.load_state_dict(prediction_model.state_dict())
    prediction_model.to(device)
    target_model.to(device)

    for _ in range(300):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        observation = processimage.process_image(observation)

        # print(_)

        print(type(observation))
        print((observation.shape))

        if terminated or truncated:
            observation, info = env.reset()

    PATH="E:\\ml\\saved_models\\trained_prediction_model.pth"
    PATH2="E:\\ml\\saved_models\\trained_target_model.pth"
    torch.save(prediction_model.state_dict(), PATH)
    torch.save(target_model.state_dict(), PATH2)
    env.close()

def dummy_function(step_count):
    print("goddamn" + str(step_count))


if __name__ == '__main__':
    dummy_main()