# 2.1 IMPORTING LIBRARIES

import sys
IN_COLAB = "google.colab" in sys.modules

import random
import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model

from IPython.display import clear_output


# 3.3 CREATING THE Q-Network
# Neural Network Model Defined at Here.
class Network(Model):
    def __init__(self, state_size: int, action_size: int, 
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.value = tf.keras.layers.Dense(action_size)

    def call(self, state):
        state = tf.convert_to_tensor(state)
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value

class DQNAgent:
    def __init__(
        self, 
        env: gym.Env,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        
        # 3.3 CREATING THE Q-Network
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.lr = 0.001
        self.gamma = 0.99
        
        self.dqn = Network(self.state_size, self.action_size
                          )
        self.optimizers = optimizers.Adam(lr=self.lr, )
        
    # 3.4.1 EXPLORATION VS EXPLOITATION
    def get_action(self, state, epsilon):
        q_value = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))[0]
        # 3. Choose an action a in the current world state (s)
        # If this number < greater than epsilon doing a random choice --> exploration
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)

        ## Else --> exploitation (taking the biggest Q value for this state)
        else:
            action = np.argmax(q_value) 

        return action
    
    # 3.4.2 UPDATING THE Q-VALUE
    def train_step(self, state, action, reward, next_state, done, next_action):
        
        dqn_variable = self.dqn.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            
            state      = np.float32(state)
            next_state = np.float32(next_state)
            
            curr_Q = self.dqn([state])[0]
            ## Obtain the Q' values by feeding the new state through our network
            next_Q = np.asarray(self.dqn([next_state]))

            ## Obtain maxQ' and set our target value for chosen action.
            q_target = np.array(curr_Q)

            # But from target model
            if done:
                q_target[action] = reward
            else:
                q_target[action] = (reward + self.gamma * next_Q[0][next_action])
                # q_target[action] = (reward + self.gamma * np.asarray(self.dqn([next_state]))[0][next_action])
            
            ## Train network using target and predicted Q values
            # it is not real target Q value, it is just an estimation,
            # but check the Q-Learning update formula:
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * Q(s',a') - Q(s,a)]
            # minimizing |r + gamma * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ~~ Q(s,a)            
            q_value = self.dqn([state])
          
            main_value   = tf.convert_to_tensor(q_value)
            target_value = tf.convert_to_tensor(q_target)
            error = tf.square(main_value - target_value) * 0.5
            loss  = tf.reduce_mean(error)
            
        dqn_grads = tape.gradient(loss, dqn_variable)
        self.optimizers.apply_gradients(zip(dqn_grads, dqn_variable))

# environment
env_name = "CartPole-v0"
env = gym.make(env_name)
env.seed(1)     # reproducible, general Policy gradient has high variance

hidden_size = 128
max_episodes = 2500  # Set total number of episodes to train agent on.

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration prob

# train
agent = DQNAgent(
    env, 
#     memory_size, 
#     batch_size, 
#     epsilon_decay,
)

if __name__ == "__main__":
    
    # 2.5 TRAINING LOOP
    #List to contain all the rewards of all the episodes given to the agent
    scores = []
    
    # 2.6 EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        
            
        # 2.7 EACH TIME STEP    
        while not done:
        # for step in range(max_steps):  # step index, maximum step is 99
        
            # 3.4.1 EXPLORATION VS EXPLOITATION
            # Take the action (a) and observe the outcome state(s') and reward (r)
            action = agent.get_action(state, epsilon)
            
            # 2.7.2 TAKING ACTION
            next_state, reward, done, _ = agent.env.step(action)
            next_action = agent.get_action(next_state, epsilon)
            
            # 3.4.2 UPDATING THE Q-VALUE
            agent.train_step(state, action, reward, next_state, done, next_action)
            
            # Our new state is state
            state = next_state

            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                break
                
        # 2.8 EXPLORATION RATE DECAY
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 


