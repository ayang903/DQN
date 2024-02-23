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
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=10000, eps_end=0.01, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=128, fc2_dims=128)

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
            # print(f"state: {state}")
            actions = self.Q_eval.forward(state) #remember, this gives out 4 outputs, take the index of biggest one
            # print(f"four actions: {actions}")
            action = T.argmax(actions).item()
            # print(f"selected action: {action}")
        else:
            action = np.random.choice(self.action_space)
            # print(f"randomly selected action: {action}")
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size: #if memory is filled up with zeros (just np.zeros like initialized), we start learning as soon as we fill up batch size of memory
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False) #grab a batch of memories

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device) # convert the numpy array subset of memory into agent's tensor
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch] 

        # now need to perform feedforward to get the parameters for loss function
        # we want to be moving the agent's estimate for the value of the current state towards the maximal value of the next state. or
        # tilt it towrads selecting maximal actions

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        # print(f"Q_EVAL: {q_eval}")
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # calculate the target values
        q_target = reward_batch + self.gamma *  T.max(q_next, dim=1)[0] #this is the purely greedy action
        # print(f"Q_TARGET: {q_target}")

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        # print(f"LOSS: {loss}")
        loss.backward()
        self.Q_eval.optimizer.step()

        # handle epsilon decrement
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


        



