import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gamelib

class ReplayBuffer():
    '''
    To store the agent's experiences network at iteration i
    Stores the state-action reward, transitions, and whether the game is done
    During learning, we apply Q-learning updates on samples of experience,
    drawn uniformly at random from the pool of stored samples
    '''
    def __init__(self, max_size, input_shape, num_actions):
        '''
        Variables:
        - memory_size: size of the replay buffer
        - discrete: whether or not our actions are deterministic
        - state_memory: records the states
        - new_state_memory: keep track of new states we get after taking an action
        - action_memory 
        - reward_memory: keep track of rewards in the environment
        - terminal_memory: keep track of terminal flags in the environment
        '''

        self.mem_size = max_size
        dtype = np.float16

        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=dtype)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=dtype)
        self.action_memory = np.zeros((self.mem_size), dtype=dtype)
        self.reward_memory = np.zeros((self.mem_size), dtype=dtype)
        self.terminal_memory = np.zeros((self.mem_size), dtype=dtype)
        self.memory_counter = 0

    def store_transition(self, state, action, reward, next_state, done):
        '''
        Store the transition function
        - index: to loop back to the start if memory is full, and replace the first experience
        '''
        index = self.memory_counter % self.mem_size
        
        # Update memory
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done) # 1 if not done, 0 if done
        self.action_memory[index] = action
        
        # Update memory_counter
        self.memory_counter += 1
        
    def sample_buffer(self, batch_size):
        '''
        Select batches of experiences from memory to perform learning
        Batch selection is random to avoid selection of sequential memories
        which would lead to correlations in learning
        '''
        max_memory = min(self.memory_counter, self.mem_size)
        batch = np.random.choice(max_memory, batch_size) # Returns an array of values, randomly chosen from max_memory

        # Get the sampled states
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminal

    def save(self, filepath):
        np.savez(filepath, 
            mem_size = self.mem_size,
            state_memory = self.state_memory,
            new_state_memory = self.new_state_memory,
            action_memory = self.action_memory,
            reward_memory = self.reward_memory,
            terminal_memory = self.terminal_memory,
            memory_counter = self.memory_counter
        )
    
    def load(self, filepath):
        npzfile = np.load(filepath)
        self.mem_size = npzfile['mem_size']
        self.state_memory = npzfile['state_memory']
        self.new_state_memory = npzfile['new_state_memory']
        self.action_memory = npzfile['action_memory']
        self.reward_memory = npzfile['reward_memory']
        self.terminal_memory = npzfile['terminal_memory']
        self.memory_counter = npzfile['memory_counter']


def build_dqn(lr, num_actions, input_shape, layer1_shape, layer2_shape):
    '''
    Build the deep Q network
    Assumes 2 hidden layers, but can be subject to change
    '''
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.Input(shape=input_shape))

    # Hidden layers
    model.add(layers.Dense(layer1_shape, activation='relu'))
    model.add(layers.Dense(layer2_shape, activation='relu'))

    # Output layer
    model.add(layers.Dense(num_actions))

    # Compile the model
    model.compile(optimizer='Adam', loss='mse')

    return model


class Agent():
    '''
    Generate the reinforcement learning agent
    '''
    def __init__(self, alpha, gamma, num_actions, epsilon, batch_size, input_shape, fname,
                 epsilon_dec=0.996, epsilon_min=0.01, memory_size=100000):
        '''
        Initialise the agent
        - alpha: learning rate for DQN
        - gamma: discount factor
        - num_actions: number of actions
        - epsilon: randomness for training (exploration)
        - batch_size: size of sample for learning
        - input_shape: shape of the environment
        - epsilon_dec: decrementing of epsilon over time
        - epsilon_min: smallest value epsilon can decrease too
        - memory_size: size of memory
        - fname: file name to save the agent
        '''
        self.action_space = [i for i in range(num_actions)]
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.model_file = fname[0]
        self.memo_file = fname[1]
        self.epsilon_file = fname[2]

        self.memory = ReplayBuffer(memory_size, input_shape, num_actions)

        # Load/Create the model and the replay buffer
        try:
          self.load_model_and_memory()
        except OSError:
          self.dqn = build_dqn(alpha, num_actions, input_shape, 128, 128)

    def decide_random_or_not(self):
        rand = np.random.random()
        return rand < self.epsilon
    
    def choose_action(self, state, random):
        '''
        Choose an action given the current state
        Exploration: performs a random action with probability epsilon
        Exploitation: chooses optimal action using DQN
        '''  
        state = state[np.newaxis, :] # Transform the variable to include 1 more dimension, to fit into the DQN
        
        # Exploration
        if random == True:
            action = np.random.choice(self.action_space)
            return action
        # Exploitation
        else:
            actions = self.dqn.predict(state)[0]
            action_indices = np.argsort(-actions) # Sort the indices of Q_value based on value in descending order
            return action_indices

    def learn(self):
        '''
        Learn by temporal difference learning (supposedly)
        Only start learning when the memory is at least of size: batch_size
        '''
        # If experiences not sufficient
        if self.memory.memory_counter < self.batch_size:
            return
        
        # Else
        # Get a batch of experiences
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        action_indices = action.astype(np.int32)
        
        # Evaluate Q-function of the states
        q_eval = self.dqn.predict(state)
        q_next = self.dqn.predict(next_state)
        q_target = q_eval.copy() # Utilities to train our DQN on

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Update our utilities with TD learning
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done

        # Fit our DQN
        self.dqn.fit(state, q_target, verbose=0)

        # Update epsilon, so DQN does not train on random actions forever
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min

    def save_model_and_memory(self):
        # Save replay buffer
        self.memory.save(self.memo_file)
        # Save epsilon value
        np.savez(self.epsilon_file, epsilon = self.epsilon)
        # Save model
        self.dqn.save(self.model_file)

    def load_model_and_memory(self):
        # Load replay buffer
        self.memory.load(self.memo_file)
        # Load epsilon value
        self.epsilon = np.load(self.epsilon_file)["epsilon"]
        # Save model
        self.dqn = tf.keras.models.load_model(self.model_file)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)