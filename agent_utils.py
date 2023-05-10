import random
import time
from collections import deque
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import os
from replaybuffer import ReplayBuffer
from dqn import DQN


class Agent:
    def __init__(self, env, reward_threshold=None):
        """Creates a DQN based agent for a given gym environment

        Args:
            reward_threshold (float): To set a reward threshold after which to stop training the agent. Defaults to the
            environment's solved threshold if not specified
            env (object): OpenAI gym environment

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
                    gamma=1, update_frequency=10, batch_size=32, clip_rewards=False, show_time=False,
                    delay_decay=False):
        """Trains the Agent using the specified parameters
            Args:
                delay_decay (bool): if True epsilon decay starts only after a positive reward has been received. Set up for the Mountain Car environment
                gamma (float): Discount rate between 0 and 1
                batch_size (int): batch_size sampled from the replay buffer over which we train
                show_time (bool): Outputs the time taken to (successfully) train the agent if True
                update_frequency (int): Number of episodes between updates of the target policy
                eps_decay (float): (1-the rate at which epsilon is decayed per episode). If set to 1, there will be no epsilon decay
                epsilon_end (float): set a value for minimum epsilon
                epsilon (float): epsilon at the start of learning
                episodes (int): Maximum number of episodes
                replay_buffer (int): The maximum number of transitions to be stored in the ReplayBuffer
                clip_rewards (bool): keeps all rewards to range (b,a)
                dqn_params (mapping): The parameters of the underlying DQNs (same parameters for policy and target networks)

            Returns:
                self.training_dict(dict): Dictionary with episode rewards and durations
            """
        # Make some assertions
        assert epsilon >= epsilon_end, "Starting epsilon should not be less that end epsilon"

        if show_time:
            strt = time.time()

        self.policy_dqn = DQN(**dqn_params)
        self.target_dqn = DQN(**dqn_params)
        self.update_target()
        self.target_dqn.eval()

        optimizer = self.policy_dqn.optim
        memory = ReplayBuffer(replay_buffer)

        episode_durations = []
        episode_rewards = []
        scores_window = deque(maxlen=100)

        print(f'Training {self.label}...')

        for i_episode in range(episodes):
            if (i_episode + 1) % (episodes / 10) == 0:
                print("episode ", i_episode + 1, "of max", episodes)

            observation, _ = self.env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            episode_reward = 0

            while not (done or terminated):

                # Select and perform an action
                #action = np.array(self.epsilon_greedy(epsilon, self.policy_dqn, state), dtype=np.float32)
                action = self.epsilon_greedy(epsilon, self.policy_dqn, state)

                print(action)
                print(type(action))

                observation, reward, done, terminated, _ = self.env.step(action)
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
                    mse_loss = self.loss(state_batch, action_batch, reward_batch, nextstate_batch, dones, gamma)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()

                if done or terminated:
                    episode_durations.append(t + 1)
                    episode_rewards.append(episode_reward)
                    scores_window.append(episode_reward)
                    # print(np.mean(scores_window))
                    if delay_decay and episode_reward > 0:
                        delay_decay = False
                        print(f'Received positive reward at episode {i_episode}.',
                              'Will begin epsilon decay now')

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
            if epsilon > epsilon_end and not delay_decay:
                epsilon *= eps_decay
                epsilon = max(epsilon_end, epsilon)

        print("Training is complete")
        if show_time:
            endt = time.time()
            self.print_time(strt, endt)

        self.training_dict = dict(episode_durations=episode_durations, episode_rewards=episode_rewards)
        return self.training_dict

    def evaluate_agent(self, episodes, plots=True, save_every=None, nb_render=False):
        """Evaluates performance of Trained Agent over a number of episodes
            Args:
                episodes(int): Number of episodes the train agent carries out
                plots (bool): Plots the score curve for the episodes if true
                save_every(int or None): x s.t. the rendering gif is saved every x episodes. So 10 means every 10th
                rendering is saved. If it is None, no rendering is saved
                nb_render(bool): Passed on to the save_render function

            Returns:
                dict: Dictionary with episode rewards and durations
            """

        episode_durations = []
        episode_rewards = []

        # Set save_every so that it is not a divisor for any number in range(episodes)
        if save_every is None:
            save_every = episodes + 2
        print("Evaluating Trained Agent...")

        # Turn off train mode
        self.policy_dqn.eval()

        for i_episode in range(episodes):
            if (i_episode + 1) % (episodes / 10) == 0:
                print("episode ", i_episode + 1, "of", episodes)

            observation, _ = self.env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            episode_reward = 0
            frames = []

            while not (done or terminated):
                frames.append(self.env.render())

                # Select and perform an action
                action = self.greedy_action(self.policy_dqn, state)

                observation, reward, done, terminated, _ = self.env.step(action)
                episode_reward += reward
                next_state = torch.tensor(observation).reshape(-1).float()

                # Move to the next state
                state = next_state

                if done or terminated:
                    episode_durations.append(t + 1)
                    episode_rewards.append(episode_reward)
                    print(f'Episode {i_episode + 1} with reward {episode_reward}')
                    print(f'{t + 1} steps')

                    if ((i_episode + 1) % save_every) == 0:
                        self.save_render(frames, i_episode, nb_render=nb_render)

                t += 1
        if plots:
            self.plot_episodes(episode_rewards)

        return dict(episode_durations=episode_durations, episode_rewards=episode_rewards)

    def save_random_renders(self, episodes=1, plots=False, save_every=1, nb_render=True):
        """Evaluates performance of Trained Agent over a number of episodes
            Args:
                episodes(int): Number of episodes the train agent carries out
                plots (bool): Plots the score curve for the episodes if true
                save_every(int or None): x s.t. the rendering gif is saved every x episodes. So 10 means every 10th
                rendering is saved. If it is None, no rendering is saved
                nb_render(bool): Passed on to the save_render function

            Returns:
                dict: Dictionary with episode rewards and durations
            """

        episode_durations = []
        episode_rewards = []

        # Set save_every so that it is not a divisor for any number in range(episodes)
        if save_every is None:
            save_every = episodes + 2
        print("saving a random render")

        # Turn off train mode
        # self.policy_dqn.eval()

        for i_episode in range(episodes):
            if (i_episode + 1) % (episodes / 10) == 0:
                print("episode ", i_episode + 1, "of", episodes)

            self.env.reset()

            done = False
            terminated = False
            t = 0
            episode_reward = 0
            frames = []

            while not (done or terminated):
                frames.append(self.env.render())

                # Select and perform a random action
                action = self.env.action_space.sample()

                _, reward, done, terminated, _ = self.env.step(action)
                episode_reward += reward
                # next_state = torch.tensor(observation).reshape(-1).float()

                # Move to the next state
                # state = next_state

                if done or terminated:
                    episode_durations.append(t + 1)
                    episode_rewards.append(episode_reward)
                    print(f'Random episode {i_episode + 1} with reward {episode_reward}')
                    print(f'{t + 1} steps')

                    if ((i_episode + 1) % save_every) == 0:
                        self.save_render(frames, i_episode, mode='rand',nb_render=nb_render)

                t += 1
        if plots:
            self.plot_episodes(episode_rewards)

        return None

    def save_render(self, frames, i_episode, mode='eval', nb_render=False):
        """
        Saves the rendering as a gif
        Args:
            frames (list(images)): list of image frames saved from rendering
            i_episode (int): the current episode of learning
            mode (str): where in 'eval' (evaluating trained agent), 'random'(evaluating untrained agent) or 'training' (saving renderings of an agent in training) mode
            nb_render (bool): if True, indicates that the rendering is for an ipynb and is saved without a timestamp.

        Returns:
            None

        """
        folder_name = 'images/' + self.env.unwrapped.spec.id
        self.check_path_exists(folder_name)

        if mode == 'eval':
            mode = 'evaluation_'
        elif mode == 'rand':
            mode = 'random_'
        else:
            mode = 'training_'

        if nb_render:
            imageio.mimsave(
                str(folder_name) + '/' + mode[:-1] + '.gif',
                frames, fps=15)

        else:
            imageio.mimsave(
                str(folder_name) + '/' + mode + str(i_episode + 1) + '_' + time.strftime("%y%m%d_%H%M") + '.gif',
                frames, fps=15)

        pass

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

        Returns:
            None
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

    def loss(self, states, actions, rewards, next_states, dones, gamma):
        """Calculate Bellman error loss

        Args:
            states (torch.tensor): Tensor of batched states
            actions (torch.tensor):Tensor of batched actions
            rewards (torch.tensor): Tensor of batched rewards
            next_states (torch.tensor): Tensor of batched next_states
            dones (torch.tensor): Tensor of batched bools, True when episode terminates
            gamma (float): Discount rate


        Returns:
            Float: scalar tensor with the loss value
        """

        bellman_targets = gamma * (~dones).reshape(-1) * (self.target_dqn(next_states)).max(1).values + rewards.reshape(
            -1)
        q_values = self.policy_dqn(states).gather(1, actions).reshape(-1)

        return ((q_values - bellman_targets) ** 2).mean()

    @staticmethod
    def clip_reward(reward, a=-1, b=1):
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
        """Calculates time difference and prints the execution time in hours, minutes and seconds

        Args:
            strt(float): the start time
            endt(float): the end time
        """
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


class DDQNAgent(Agent):
    def __init__(self, env, reward_threshold=None):
        """Initialize the DDQN Agent

        Args:
            env (gym.env): Gym environment
            reward_threshold (float): Reward threshold for the environment
            """

        super().__init__(env=env, reward_threshold=reward_threshold)
        self.label = 'DDQN Agent'

    def loss(self, states, actions, rewards, next_states, dones, gamma):
        """Calculate Bellman error loss

        Args:
            states (torch.tensor): Tensor of batched states
            actions (torch.tensor):Tensor of batched actions
            rewards (torch.tensor): Tensor of batched rewards
            next_states (torch.tensor): Tensor of batched next_states
            dones (torch.tensor): Tensor of batched bools, True when episode terminates
            gamma (float): Discount rate


        Returns:
            Float: scalar tensor with the loss value
        """

        # The below code first determines the ideal actions using the policy network,
        # and then computes their Q Values using the target network

        policy_dqn_actions = self.policy_dqn(next_states).max(1).indices.reshape([-1, 1])
        q_vals = self.target_dqn(next_states).gather(1, policy_dqn_actions).reshape(-1)
        bellman_targets = gamma * (~dones).reshape(-1) * q_vals + rewards.reshape(-1)
        q_values = self.policy_dqn(states).gather(1, actions).reshape(-1)

        return ((q_values - bellman_targets) ** 2).mean()


def dqn_example(gym_env):
    dqn_agent = Agent(gym_env)

    input_size = gym_env.observation_space.shape[0]
    output_size = gym_env.action_space.n

    # DQN Parameters
    layers = [input_size, 256, 128, output_size]
    activation = 'relu'
    weights = 'xunif'
    optim = 'Adam'
    learning_rate = 5e-4
    dqn_params = dict(layers=layers, activation=activation, weights=weights, optim=optim, learning_rate=learning_rate)

    # Training Parameters
    epsilon = 1
    eps_decay = 0.995
    replay_buffer = 100000
    batch_size = 128
    epsilon_end = 0.01
    episodes = 100
    update_frequency = 5
    clip_rewards = False

    training_params = dict(epsilon=epsilon, eps_decay=eps_decay, replay_buffer=replay_buffer,
                           batch_size=batch_size, epsilon_end=epsilon_end, episodes=episodes,
                           update_frequency=update_frequency, dqn_params=dqn_params, clip_rewards=clip_rewards)

    run_stats = dqn_agent.train_agent(show_time=True, **training_params)
    dqn_agent.plot_episodes(run_stats['episode_rewards'])


def ddqn_example(gym_env):
    """Example of how to use the DDQN Agent to solve an environment"""
    ddqn_agent = DDQNAgent(gym_env)

    input_size = gym_env.observation_space.shape[0]
    output_size = gym_env.action_space.n

    # DDQN Parameters
    layers = [input_size, 256, 128, output_size]  # DDQN Architecture
    activation = 'relu'
    weights = 'xunif'
    optim = 'Adam'
    learning_rate = 5e-4
    dqn_params = dict(layers=layers, activation=activation, weights=weights, optim=optim, learning_rate=learning_rate)

    # Training Parameters
    epsilon = 1
    eps_decay = 0.995
    replay_buffer = 100000
    batch_size = 64
    epsilon_end = 0.01
    episodes = 1000
    update_frequency = 5
    clip_rewards = False

    training_params = dict(epsilon=epsilon, eps_decay=eps_decay, replay_buffer=replay_buffer,
                           batch_size=batch_size, epsilon_end=epsilon_end, episodes=episodes,
                           update_frequency=update_frequency, dqn_params=dqn_params, clip_rewards=clip_rewards)

    run_stats = ddqn_agent.train_agent(show_time=True, **training_params)
    ddqn_agent.plot_episodes(run_stats['episode_rewards'])


if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode='rgb_array')
    dqn_example(gym_env=env)
    # ddqn_example(gym_env=env)
