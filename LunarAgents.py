import numpy as np
import pandas as pd
import gym


class DisAgent:
    def __init__(self, env, grid_w=10):
        """

        :type grid_w: int
        """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observartion_space
        self.grid_h = self.observation_space.shape[0]
        self.grid_w = grid_w
        self.policy, self.Q = self.initialise()

    def initialise(self):
        """Makes initialise policy and initial estimate of state action value function"""
        policy_grid = (1 / self.grid_w) * np.ones(shape=(self.grid_h, self.grid_w))
        Q = np.random.uniform(size=(self.grid_h, self.grid_w))

        def policy_fn(observation):
            pass

        return policy_fn, Q




class MCAgent(DisAgent):
    def __init__(self, env, grid_w=10):
        """

        :type env: object
        """
        super().__init__(env, grid_w)


if __name__ == '__main__':
    #env = gym.make("MountainCar-v0", render_mode="human")

    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)
    for _ in range(10):
        # action = policy(observation)  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())


        if terminated or truncated:
            observation, info = env.reset()
    env.close()

