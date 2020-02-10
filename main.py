import gym
from functions import play_one_episode
from classes import Agent
from datetime import datetime

NUM_EPISODES = 30000
BATCH_SIZE = 32

env = gym.make('LunarLander-v2')

state_size = env.observation_space.shape[0]
action_size = 4

agent = Agent(state_size, action_size, BATCH_SIZE)

for e in range(NUM_EPISODES):
    t0 = datetime.now()

    total_reward = play_one_episode(env, agent)
    dt = datetime.now() - t0
    print("episode: {0}/{1}, duration: {2}".format(e + 1, NUM_EPISODES, dt))
    print("total reward: {0}".format(total_reward))