# IMPORTING LIBRARIES

import sys
IN_COLAB = "google.colab" in sys.modules

import random
import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from collections import deque

from IPython.display import clear_output

import collections

class SumTree:
    data_pointer = 0
    
    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
    
    # Here we define function that will add our priority score in the sumtree leaf and add the experience in data:
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        
        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update (tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        self._propagate(tree_index, change)

    def _retrieve(self, idx, s):
        left_child_index = 2 * idx + 1
        right_child_index = left_child_index + 1
        if left_child_index >= len(self.tree):
            return idx
        if s <= self.tree[left_child_index]:
            return self._retrieve(left_child_index, s)
        else:
            return self._retrieve(right_child_index, s - self.tree[left_child_index])

    def get_leaf(self, s):
        leaf_index = self._retrieve(0, s)

        data_index = leaf_index - self.capacity + 1

        return (leaf_index, self.tree[leaf_index], self.data[data_index])
    
    def total_priority(self):
        return self.tree[0] # Returns the root node

# Now we finished constructing our SumTree object, next we'll build a memory object.
class PrioritizedReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    PER_e = 0.001
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    

    def __init__(self, capacity):
        # Making the tree 
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.PER_e) ** self.PER_a

    def store(self, error, sample):
        max_priority = self._getPriority(error)
        self.tree.add(max_priority, sample)
        
    # Now we create sample function, which will be used to pick batch from our tree memory, which will be used to train our model.
    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then a value is uniformly sampled from each range.
    # - Then we search in the sumtree, for the experience where priority episode_reward correspond to sample values are retrieved from.
    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []
        idxs = []
        priority_segment = self.tree.total_priority() / n
        priorities = []
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        for i in range(n):
            # A value is uniformly sample from each range
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            (idx, p, data) = self.tree.get_leaf(value)
            priorities.append(p)
            minibatch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total_priority()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.PER_b)
        is_weight /= is_weight.max()

        return minibatch, idxs, is_weight

    def batch_update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

# CREATING THE Dueling Q-Network
# Neural Network Model Defined at Here.
class Network(Model):
    def __init__(self, state_size: int, action_size: int, 
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.num_action = action_size
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.state = tf.keras.layers.Dense(self.num_action)
        self.action = tf.keras.layers.Dense(self.num_action)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        state = self.state(layer2)
        action = self.action(layer2)
        mean = tf.keras.backend.mean(action, keepdims=True)
        advantage = (action - mean)
        value = state + advantage        
        return value

class DQNAgent:
    def __init__(
        self, 
        env: gym.Env,
        batch_size: int,
        target_update: int,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        
        # CREATING THE Q-Network
        self.env = env
        self.env.seed(0)  
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.batch_size = batch_size
        # hyper parameters
        memory_size = 10000
        self.lr = 0.001
        self.target_update = target_update
        self.gamma = 0.99
        
        self.dqn = Network(self.state_size, self.action_size
                          )
        self.dqn_target = Network(self.state_size, self.action_size
                          )
        self.train_start = 1000

        self.optimizers = optimizers.Adam(lr=self.lr, )
        
        self.MEMORY = PrioritizedReplayBuffer(memory_size)
        self.Soft_Update = False # use soft parameter update

        self.TAU = 0.1 # target network soft update hyperparameter
        
        self._target_hard_update()
        
    # EXPLORATION VS EXPLOITATION
    def get_action(self, state, epsilon):
        q_value = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
        # 3. Choose an action a in the current world state (s)
        # If this number < greater than epsilon doing a random choice --> exploration
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)

        ## Else --> exploitation (taking the biggest Q value for this state)
        else:
            action = np.argmax(q_value) 

        return action
    
    def append_sample(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)

        main_next_q = np.array(self.dqn(next_state))[0]
        next_action = np.argmax(main_next_q)
        target_next_q = np.array(self.dqn_target(next_state))[0]
        target_value = target_next_q[next_action]

        target_value = target_value * 0.99 * (1-done) + reward

        main_q = np.array(self.dqn(state))[0]
        main_q = main_q[action]

        td_error = np.abs(target_value - main_q)

        self.MEMORY.store(td_error, (state, action, reward, next_state, done))

    # UPDATING THE Q-VALUE
    def train_step(self):
        mini_batch, idxs, IS_weight = self.MEMORY.sample(self.batch_size)
        mini_batch = np.array(mini_batch)
        states      = [i[0] for i in mini_batch]
        actions     = [i[1] for i in mini_batch]
        rewards     = [i[2] for i in mini_batch]
        next_states = [i[3] for i in mini_batch]
        dones       = [i[4] for i in mini_batch]
        
        dqn_variable = self.dqn.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            
            states      = tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)
            actions     = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards     = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32)
            dones       = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            next_Qs = self.dqn(next_states)
            next_Qs = tf.stop_gradient(next_Qs)
            next_Q_targs = self.dqn_target(next_states)
            next_action = tf.argmax(next_Qs, axis=1)
            target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * next_Q_targs, axis=1)
            
            mask = 1 - dones
            target_value = rewards + self.gamma * target_value * mask 
            
            curr_Qs = self.dqn(states)
            
            main_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * curr_Qs, axis=1)
            error = tf.square(main_value - target_value) * 0.5
            error = error * tf.convert_to_tensor(IS_weight, dtype=tf.float32)
            loss  = tf.reduce_mean(error)
            
        dqn_grads = tape.gradient(loss, dqn_variable)
        self.optimizers.apply_gradients(zip(dqn_grads, dqn_variable))
        
        state_value = np.array(self.dqn(states))
        state_value = np.array([sv[a] for a, sv in zip(np.array(actions), state_value)])

        td_error = np.abs(target_value - state_value)
        
        for i in range(self.batch_size):
            tree_idx = idxs[i]
            self.MEMORY.batch_update(tree_idx, td_error[i])
        
    # after some time interval update the target model to be same with model
    def _target_hard_update(self):
        if not self.Soft_Update:
            self.dqn_target.set_weights(self.dqn.get_weights())
            return
        if self.Soft_Update:
            q_model_theta = self.dqn.get_weights()
            dqn_target_theta = self.dqn_target.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, dqn_target_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                dqn_target_theta[counter] = target_weight
                counter += 1
            self.dqn_target.set_weights(dqn_target_theta)
    
    def load(self, name):
        self.dqn = load_model(name)

    def save(self, name):
        self.dqn.save(name)
    
# CREATING THE ENVIRONMENT
env_name = "CartPole-v0"
env = gym.make(env_name)

# parameters
target_update = 20


# INITIALIZING THE Q-PARAMETERS
hidden_size = 64
max_episodes = 300  # Set total number of episodes to train agent on.
batch_size = 64

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.025            # Exponential decay rate for exploration prob

# train
agent = DQNAgent(
    env, 
#     memory_size, 
    batch_size, 
    target_update, 
#     epsilon_decay,
)

if __name__ == "__main__":
    
    update_cnt    = 0
    # TRAINING LOOP
    #List to contain all the rewards of all the episodes given to the agent
    scores = []
    
    # EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        
            
        # EACH TIME STEP    
        while not done:
        # for step in range(max_steps):  # step index, maximum step is 200
            update_cnt += 1
            # 3.4.1 EXPLORATION VS EXPLOITATION
            # Take the action (a) and observe the outcome state(s') and reward (r)
            action = agent.get_action(state, epsilon)
            
            # 2.7.2 TAKING ACTION
            next_state, reward, done, _ = agent.env.step(action)
            agent.append_sample(state, action, reward, next_state, done)
            
            # Our new state is state
            state = next_state

            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("episode: {}/{}, score: {}, e: {:.4}".format(episode+1, max_episodes, episode_reward, epsilon)) 
                break
            # if training is ready
            if (update_cnt >= agent.batch_size):
                # 3.4.2 UPDATING THE Q-VALUE
                agent.train_step()

            
                # if hard update is needed
                if update_cnt % agent.target_update == 0:
                    agent._target_hard_update()
            
        # 2.8 EXPLORATION RATE DECAY
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 

        
