import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
import gym_examples

# Assuming you have the GridWorldEnv class defined elsewhere in your script

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor
        self.lr = 5e-4
        self.batch_size = 64
        self.memory = ReplayBuffer(10000)
        
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to evaluation mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return np.argmax(action_values.cpu().data.numpy())
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from policy net
        Q_expected = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        print(f"LOSS:{loss}")
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training Loop
def train_dqn(env):
    state_size = 16
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000  # Number of episodes to train
    
    for e in range(episodes):
        observation, _ = env.reset()  # Reset returns an observation and info
        state = observation['agent']  # Assuming the observation is a dictionary with the agent's position
        # No need to reshape state here since it should already be in the correct format (16,)
        done = False
        total_loss = 0
        time_step = 0

        while not done:
            action = agent.act(torch.FloatTensor(state).unsqueeze(0))  # Convert state to tensor and add batch dimension
            observation, reward, done, truncated, _ = env.step(action)
            next_state = observation['agent']  # Update next_state with the agent's new position

            agent.memory.push(state, action, reward, next_state, done)
            state = next_state  # Update state for the next iteration
            loss = agent.learn()
            total_loss += loss
            time_step += 1

        if e % 10 == 0:
            agent.update_target_network()
            print(f"Episode {e}/{episodes} - Time step: {time_step} - Average Loss: {total_loss/time_step if time_step else 0:.4f} - Epsilon: {agent.epsilon:.2f}")


env = gym.make('gym_examples/GridWorld-v0', size=4)
train_dqn(env)
