# 2.1 IMPORTING LIBRARIES

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
        
        # 2.3 CREATING THE Q-TABLE
        self.env = env
        
        self.state_size  = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        
        self.lr = 0.9
        self.gamma = 0.99
        
        self.qtable = np.zeros((self.state_size, self.action_size))
        print(self.qtable)
        
    # 2.7.1 EXPLORATION VS EXPLOITATION
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
    
    # 2.7.3 UPDATING THE Q-VALUE
    def train_step(self, state, action, reward, next_state, done):
        
        curr_Q = self.qtable[state, action]
        next_Q = self.qtable[next_state, :]

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a) - Q(s,a)]
        # qtable[next_state,:] : all the actions we can take from new state
        # self.qtable[state, action] = self.qtable[state, action] + self.lr * (reward + gamma * np.max(self.qtable[next_state, :]) - self.qtable[state, action])
        self.qtable[state, action] = curr_Q + self.lr * (reward + gamma * np.max(next_Q) - curr_Q)

# 2.2 CREATING THE ENVIRONMENT
env_name = "FrozenLake-v1"
env = gym.make(env_name)
env.seed(777)     # reproducible, general Policy gradient has high variance

# 2.4 INITIALIZING THE Q-PARAMETERS
max_episodes = 5000  # Set total number of episodes to train agent on.

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
    
    # 2.5 TRAINING LOOP
    #List to contain all the rewards of all the episodes given to the agent
    scores = []
    
    # 2.6 EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        
        if render: env.render()
            
        # 2.7 EACH TIME STEP    
        while not done:
        # for step in range(max_steps):  # step index, maximum step is 99
        
            # 2.7.1 EXPLORATION VS EXPLOITATION
            # Take the action (a) and observe the outcome state(s') and reward (r)
            action = agent.get_action(state, epsilon)
            
            # 2.7.2 TAKING ACTION
            next_state, reward, done, _ = agent.env.step(action)

            if render: env.render()
                
            # 2.7.3 UPDATING THE Q-VALUE
            agent.train_step(state, action, reward, next_state, done)
            
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

    print ("Score over time: " +  str(sum(scores)/max_episodes))
    print(agent.qtable)
    
    # Calculate and print the average reward per thousand episodes
    # rewards_per_thousand_episodes = np.split(np.array(scores),int(max_episodes/1000), axis=0)
    count = 1000
    rewards_per_thousand_episodes = np.split(np.array(scores),int(max_episodes/count))

    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000
    #Print the updates Q-Table
    print("\n\n*******Q-Table*******\n")
    print(agent.qtable)


    # Replay the taining result
    ! pip install gym[toy_text]
    
    # Watch our agent play Frozen Lake by playing the best action 
    # from each state according to the Q-table    agent.env.reset()

    for episode in range(5):
        state = agent.env.reset()
        step = 0
        done = False
        print("****************************************************")
        print("*******Episode ", episode+1, "*******\n\n\n\n")
        time.sleep(1)

        for step in range(max_steps):
            # Show current state of environment on screen
            clear_output(wait=True)
            # Choose action with highest Q-value for current state
            agent.env.render()
            # Take new action
            time.sleep(0.3)
            
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(agent.qtable[state,:])

            new_state, reward, done, info = agent.env.step(action)
            
            if done:
                # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                if new_state == 15:
                    print("We reached our Goal 🏆")
                else:
                    print("We fell into a hole ☠️")

                # We print the number of step it took.
                print("Number of steps", step)

                break
            state = new_state
    agent.env.close()
    