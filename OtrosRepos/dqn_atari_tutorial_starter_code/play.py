import gym
from gym.utils.play import play

env = gym.make('Breakout-v4', render_mode = "rgb_array")

play(env, zoom=3)
