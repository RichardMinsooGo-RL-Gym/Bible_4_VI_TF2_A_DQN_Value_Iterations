import sys
IN_COLAB = "google.colab" in sys.modules

import tensorflow as tf
import random
import gym
import numpy as np

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from collections import deque

from IPython.display import clear_output

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
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value

class DQNAgent:
    def __init__(
        self, 
        env: gym.Env,
        batch_size: int,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.batch_size = batch_size
        # hyper parameters
        self.lr = 0.001
        self.gamma = 0.99

        self.dqn = Network(self.state_size, self.action_size
                          )
        self.optimizers = optimizers.Adam(lr=self.lr, )

        self.memory = deque(maxlen=2000)

    def get_action(self, state, epsilon):
        q_value = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))[0]
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value) 
        return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        mini_batch = random.sample(self.memory, self.batch_size)

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
            
            curr_Qs    = self.dqn(states)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * curr_Qs, axis=1)
            next_Qs = self.dqn(next_states)
            next_action = tf.argmax(next_Qs, axis=1)
            target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * next_Qs, axis=1)
            
            mask = 1 - dones
            target_value = rewards + self.gamma * target_value * mask 

            error = tf.square(main_value - target_value) * 0.5
            loss  = tf.reduce_mean(error)
            
        dqn_grads = tape.gradient(loss, dqn_variable)
        self.optimizers.apply_gradients(zip(dqn_grads, dqn_variable))

# environment
env_name = "CartPole-v0"
env = gym.make(env_name)
env.seed(1)     # reproducible, general Policy gradient has high variance

hidden_size = 128
max_episodes = 200  # Set total number of episodes to train agent on.
batch_size = 64

# train
agent = DQNAgent(
    env, 
#     memory_size, 
    batch_size, 
#     epsilon_decay,
)

if __name__ == "__main__":
    
    update_cnt    = 0
    scores        = []
    
    for episode in range(max_episodes):
        state = agent.env.reset()
        episode_reward = 0
        done = False
        
        epsilon = 1 / (episode * 0.1 + 1)
        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, _ = agent.env.step(action)
            agent.append_sample(state, action, reward, next_state, done)

            
            state = next_state
            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                
            # if training is ready
            if (len(agent.memory) >= agent.batch_size):
                agent.train_step()
                update_cnt += 1

