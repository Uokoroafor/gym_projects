import random
import time
from collections import deque
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import os
from ReplayBuffer import ReplayBuffer
from DQN import DQN


class Agent:
    def __init__(self, env, reward_threshold=None):
        """Object for a DQN based agent

        Args:
            env (object): 
        """
        self.trained_state_dict = None
        self.training_dict = None
        self.policy_dqn = None
        self.env = env
        self.target_dqn = None
        self.label = 'DQN Agent'
        if reward_threshold is None:
            self.threshold = self.env.spec.reward_threshold
        else:
            self.threshold = reward_threshold

    def train_agent(self, dqn_params, replay_buffer, episodes, epsilon, epsilon_end=0.01, eps_decay=1,
                    update_frequency=10, batch_size=32, clip_rewards=False, show_time=False):
        """Trains the dqn using the DQN parameters
            Args:
                show_time (bool):
                update_frequency (int):
                eps_decay (float):
                epsilon_end (float):
                epsilon (float):
                episodes (int):
                replay_buffer (int):
                clip_rewards (bool):
                dqn_params (mapping):

            Returns:
                self.training_dict(dict): Dictionary with episode rewards and durations
            """
        if show_time:
            strt = time.time()

        self.policy_dqn = DQN(**dqn_params)
        self.target_dqn = DQN(**dqn_params)
        self.update_target()
        self.target_dqn.eval()
        # print(self.policy_dqn.state_dict())

        optimizer = self.policy_dqn.optim
        memory = ReplayBuffer(replay_buffer)

        episode_durations = []
        episode_rewards = []
        scores_window = deque(maxlen=100)

        print('Training Agent...')

        for i_episode in range(episodes):
            if (i_episode + 1) % (episodes / 10) == 0:
                print("episode ", i_episode + 1, "of max", episodes)

            observation, info = self.env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            episode_reward = 0

            while not (done or terminated):

                # Select and perform an action
                action = self.epsilon_greedy(epsilon, self.policy_dqn, state)

                observation, reward, done, terminated, info = self.env.step(action)
                episode_reward += reward
                if clip_rewards:
                    reward = self.clip_reward(reward)

                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                memory.push([state, action, next_state, reward, torch.tensor([done])])
                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if not len(memory.buffer) < batch_size:
                    transitions = memory.sample(batch_size)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in
                                                                                       zip(*transitions))
                    # Compute loss
                    mse_loss = self.loss(state_batch, action_batch, reward_batch, nextstate_batch, dones)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()

                if done or terminated:
                    episode_durations.append(t + 1)
                    episode_rewards.append(episode_reward)
                    scores_window.append(episode_reward)

                t += 1

            # Update the target dqn, copying all weights and biases in DQN
            if i_episode % update_frequency == 0:
                self.update_target()

            # Check if solved threshold has been reached
            if np.mean(scores_window) >= self.threshold:
                print(f'Environment solved within {i_episode + 1} episodes.')
                print(f'Average Score: {np.mean(scores_window)}')
                break

            # Update epsilon
            if epsilon > epsilon_end:
                epsilon *= eps_decay
                epsilon = max(epsilon_end, epsilon)

        print("Training is complete")
        if show_time:
            endt = time.time()
            self.print_time(strt, endt)

        self.training_dict = dict(episode_durations=episode_durations, episode_rewards=episode_rewards)
        return self.training_dict

    def evaluate_agent(self, episodes, plots=True, save_every=1):
        """Evaluates performance of Trained Agent over a number of episodes
            Returns:
                Greedy action according to DQN
            """

        episode_durations = []
        episode_rewards = []
        # print(self.policy_dqn.state_dict())
        print("Evaluating Trained Agent...")

        self.policy_dqn.eval()
        for i_episode in range(episodes):
            if (i_episode + 1) % (episodes / 10) == 0:
                print("episode ", i_episode + 1, "of", episodes)

            observation, info = self.env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            episode_reward = 0
            frames = []
            rewards = []

            while not (done or terminated):
                frames.append(self.env.render())

                # Select and perform an action

                action = self.greedy_action(self.policy_dqn, state)

                observation, reward, done, terminated, info = self.env.step(action)
                episode_reward += reward
                rewards.append(reward)
                next_state = torch.tensor(observation).reshape(-1).float()

                # Move to the next state
                state = next_state

                if done or terminated:
                    episode_durations.append(t + 1)
                    episode_rewards.append(episode_reward)
                    print(f'Episode {i_episode + 1} with reward {episode_reward}')
                    print(f'{t + 1} steps')

                    if ((i_episode + 1) % save_every) == 0:
                        folder_name = 'images/' + self.env.unwrapped.spec.id
                        self.check_path_exists(folder_name)

                        imageio.mimsave(str(folder_name) + '/' + time.strftime("%y%m%d_%H%M") + 'evaluations_' + str(
                            i_episode + 1) + '.gif', frames,
                                        fps=15)

                t += 1
        if plots:
            self.plot_episodes(episode_rewards)

        return dict(episode_durations=episode_durations, episode_rewards=episode_rewards)

    @staticmethod
    def check_path_exists(path):
        """ Checks whether the specified path exists and creates it if not"""
        dir_exist = os.path.exists(path)
        if not dir_exist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print(f"The new directory, {path}, has been created!")

    def update_target(self):
        """Update target network parameters using policy network.
        """

        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

    @staticmethod
    def greedy_action(dqn, state):
        """Select action according to a given DQN

        Args:
            dqn: the DQN that selects the action
            state: state at which the action is chosen

        Returns:
            Greedy action according to DQN
        """
        return int(torch.argmax(dqn(state)))

    def epsilon_greedy(self, epsilon, dqn, state):
        """Sample an epsilon-greedy action according to a given DQN

        Args:
            epsilon: parameter for epsilon-greedy action selection
            dqn: the DQN that selects the action
            state: state at which the action is chosen

        Returns:
            epsilon-greedy action
        """
        q_values = dqn(state)
        num_actions = q_values.shape[0]
        p = random.random()
        if p > epsilon:
            return self.greedy_action(dqn, state)
        else:
            return random.randint(0, num_actions - 1)

    def loss(self, states, actions, rewards, next_states, dones, is_DQN=True):
        """Calculate Bellman error loss

        Args:
            states: batched state tensor
            actions: batched action tensor
            rewards: batched rewards tensor
            next_states: batched next states tensor
            dones: batched Boolean tensor, True when episode terminates
            is_DQN (bool): True if policy is a DQN

        Returns:
            Float scalar tensor with loss value
        """
        if is_DQN:
            bellman_targets = (~dones).reshape(-1) * (self.target_dqn(next_states)).max(1).values + rewards.reshape(-1)
            q_values = self.policy_dqn(states).gather(1, actions).reshape(-1)
        elif not is_DQN:
            # The above code first determines the ideal actions using the policy network,
            # and then computes their Q Values using the target network
            policy_dqn_actions = self.policy_dqn(next_states).max(1).indices.reshape([-1, 1])
            q_vals = self.target_dqn(next_states).gather(1, policy_dqn_actions).reshape(-1)
            bellman_targets = (~dones).reshape(-1) * q_vals + rewards.reshape(-1)
            q_values = self.policy_dqn(states).gather(1, actions).reshape(-1)

        return ((q_values - bellman_targets) ** 2).mean()
        # return F.mse_loss(q_values, bellman_targets)

    @staticmethod
    def clip_reward(reward, a=-10, b=10):
        if reward < a:
            return a
        elif reward > b:
            return b
        else:
            return reward

    def plot_episodes(self, episode_stats):
        rewards = torch.tensor(episode_stats)
        means = rewards.float()
        # stds = rewards.float().std(0)

        plt.plot(torch.arange(len(means)), means, label=self.label, color='g')
        plt.ylabel("score")
        plt.xlabel("episode")
        # plt.fill_between(np.arange(episodes), means, means + stds, alpha=0.3, color='g')
        # plt.fill_between(np.arange(episodes), means, means - stds, alpha=0.3, color='g')
        plt.axhline(y=self.threshold, color='r', linestyle='dashed', label='Solved Threshold')
        plt.legend()
        plt.show()

    @staticmethod
    def print_time(strt, endt):
        # get the execution time
        elapsed_ = endt - strt
        hours = elapsed_ // (60 * 60)
        minutes = (elapsed_ % (60 * 60)) // 60
        seconds = elapsed_ % 60
        str_time = ''
        if hours > 0:
            str_time += str(hours) + ' hours, '
        if minutes > 0:
            str_time += str(minutes) + ' minutes, '
        str_time += str(seconds) + ' seconds.'
        print('Execution time: ' + str_time)

class DDQN_Agent(Agent):
    def __init__(self):
        pass

def dqn_example():
    env = gym.make("LunarLander-v2", render_mode='rgb_array')
    dqn_agent = Agent(env)

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # DQN Parameters
    layers = [input_size, 128, 128, output_size]  # DQN Architecture
    activation = 'relu'
    weights = 'xunif'
    optim = 'Adam'
    learning_rate = 5e-4
    dqn_params = dict(layers=layers, activation=activation, weights=weights, optim=optim, learning_rate=learning_rate)

    # Training Parameters
    epsilon = 1
    eps_decay = 0.995  # Epsilon is reduced by 1-eps_decay every episode
    replay_buffer = 100000
    batch_size = 64
    epsilon_end = 0.01
    episodes = 100
    update_frequency = 5
    clip_rewards = False

    training_params = dict(epsilon=epsilon, eps_decay=eps_decay, replay_buffer=replay_buffer,
                           batch_size=batch_size, epsilon_end=epsilon_end, episodes=episodes,
                           update_frequency=update_frequency, dqn_params=dqn_params, clip_rewards=clip_rewards)

    run_stats = dqn_agent.train_agent(show_time=True, **training_params)
    dqn_agent.plot_episodes(run_stats['episode_rewards'])
    # dqn_agent.evaluate_agent(10, plots=True, save_every=10)


if __name__ == '__main__':
    dqn_example()

    """rewards = torch.tensor(run_stats['episode_rewards'])
    means = rewards.float()
    #stds = rewards.float()

    plt.plot(torch.arange(episodes), means, label=dqn_agent.label, color='g')
    plt.ylabel("score")
    plt.xlabel("episode")
    #plt.fill_between(np.arange(episodes), means, means + stds, alpha=0.3, color='g')
    #plt.fill_between(np.arange(episodes), means, means - stds, alpha=0.3, color='g')
    plt.axhline(y=dqn_agent.threshold, color='r', linestyle='dashed', label='Solved')
    plt.legend()
    plt.show()"""
