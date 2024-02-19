import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = F.mse_loss
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions
    
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, update_freq, max_mem_size=10000, eps_end=0.01, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.update_freq = update_freq
        
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=128, fc2_dims=128)
        self.Q_target = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=128, fc2_dims=128)


        # something for storing memory deq
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) #remmeber, temperal diff wants state, next state, reward, action 
        # DeepQLearning is a model free, bootstrapped, off policy learning method
        # dont need to know anything about dynamics of environment, we'll figure it out by playing the game (modelfree)
        # bootstrapped =   going to construct estimates of Q function, based on earlier estimates. using one estimate to update the next
        # offpolicy = policy used to generate actions is epsilon greedy. 
                    # epsilon determines portion of time used to take random vs greedy actions. use this policy to generate data to update the purely greedy policy
                    # ^ the agents estimate of the maximum value functoin
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.terminal_memory = np.zeros(self.mem_size, dtype=bool) # if you encounter terminal state, game is done

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size # posiiton of first unoccupied memory
        # print(f"index: {index}")
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon: #take the best known action
            state = T.tensor(observation, dtype=T.float32).to(self.Q_eval.device) #take our current state (observation), turn into tensor, send to device
            print(f"state: {state}")
            actions = self.Q_eval.forward(state) #remember, this gives out 4 outputs, take the index of biggest one
            print(f"four actions: {actions}")
            action = T.argmax(actions).item()
            print(f"selected action: {action}")
        else:
            action = np.random.choice(self.action_space)
            print(f"randomly selected action: {action}")
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next_eval = self.Q_eval.forward(new_state_batch)  # Use the policy network to select action
        q_next_target = self.Q_target.forward(new_state_batch)  # Use the target network to evaluate action
        
        max_actions = T.argmax(q_next_eval, dim=1)
        q_next = q_next_target[batch_index, max_actions]
        
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next
        
        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
        # Optionally, update the target network here by hard copying or soft updating the weights
        if self.mem_cntr % self.update_freq == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())




        



