import sys
IN_COLAB = "google.colab" in sys.modules

import random
import gym
import numpy as np


from IPython.display import clear_output

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
        self.env = env
        
        self.state_size  = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        
        self.lr = 0.9
        self.gamma = 0.99
        
        self.qtable = np.zeros((self.state_size, self.action_size))
        print(self.qtable)

    def get_action(self, state, epsilon):
        q_value = self.qtable[state,:]
        # 3. Choose an action a in the current world state (s)
        # If this number < greater than epsilon doing a random choice --> exploration
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)

        ## Else --> exploitation (taking the biggest Q value for this state)
        else:
            action = np.argmax(q_value) 

        return action

    def train_step(self, state, action, reward, next_state, done):
        
        curr_Q = self.qtable[state, action]
        next_Q = self.qtable[next_state, :]

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a) - Q(s,a)]
        # qtable[next_state,:] : all the actions we can take from new state
        # self.qtable[state, action] = self.qtable[state, action] + self.lr * (reward + gamma * np.max(self.qtable[next_state, :]) - self.qtable[state, action])
        self.qtable[state, action] = curr_Q + self.lr * (reward + gamma * np.max(next_Q) - curr_Q)

# environment
env_name = "FrozenLake-v1"
env = gym.make(env_name)
env.seed(777)     # reproducible, general Policy gradient has high variance

max_episodes = 2500  # Set total number of episodes to train agent on.

max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate
render = False                # display the game environment

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
    
    # List of rewards
    scores = []
    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        
        if render: env.render()
        for step in range(max_steps):  # step index, maximum step is 99
            # Take the action (a) and observe the outcome state(s') and reward (r)
            action = agent.get_action(state, epsilon)
            next_state, reward, done, _ = agent.env.step(action)

            if render: env.render()
            
            agent.train_step(state, action, reward, next_state, done)
                
            # Our new state is state
            state = next_state

            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 

    print ("Score over time: " +  str(sum(scores)/max_episodes))
    print(agent.qtable)
    
    # Replay the taining result
    ! pip install gym[toy_text]
    
    agent.env.reset()

    for episode in range(5):
        state = agent.env.reset()
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(max_steps):

            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(agent.qtable[state,:])

            new_state, reward, done, info = agent.env.step(action)

            if done:
                # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                agent.env.render()
                if new_state == 15:
                    print("We reached our Goal 🏆")
                else:
                    print("We fell into a hole ☠️")

                # We print the number of step it took.
                print("Number of steps", step)

                break
            state = new_state
    agent.env.close()
    