import random
import time
from collections import deque
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.optim import Adam, SGD, Adagrad
from torch.utils.data import DataLoader, TensorDataset
import imageio


class DQN(nn.Module):
    def __init__(self, activation, layers, weights='xunif', optim='Adam', learning_rate=1e-3):
        super().__init__()
        self.layers = layers
        assert len(self.layers) >= 2, "There needs to be at least an input and output "
        self.layer_list = []
        self.learning_rate = learning_rate

        # Make activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sig':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        # Make weights function
        if weights == 'xunif':
            self.weights_init = nn.init.xavier_uniform_
        elif weights == 'xnorm':
            self.weights_init = nn.init.xavier_normal_
        elif weights == 'unif':
            self.weights_init = nn.init.uniform_
        else:
            self.weights_init = nn.init.normal_

        # Make Layers
        self.nn_model = self.apply_layers()

        # Make optimisation function
        if optim == 'Adam':
            self.optim = Adam
        elif optim == 'SGD':
            self.optim = SGD
        elif optim == 'Adagrad':
            self.optim = Adagrad

        self.optim = self.optim(self.parameters(), lr=self.learning_rate)

        # Make Optimisation Function

    def make_weights_bias(self, layer):
        # Initialise weights randomly and set biases to zero
        self.weights_init(layer.weight)
        nn.init.zeros_(layer.bias)

    def apply_layers(self):
        input_layer = nn.Linear(in_features=self.layers[0], out_features=self.layers[1])
        self.make_weights_bias(input_layer)
        layer_list = [input_layer]

        if len(self.layers) > 2:
            for k in range(1, len(self.layers) - 1):
                layer = nn.Linear(in_features=self.layers[k], out_features=self.layers[k + 1])
                self.make_weights_bias(layer)
                layer_list.append(self.activation)
                layer_list.append(layer)

        return nn.Sequential(*layer_list)

    def forward(self, input):

        return self.nn_model(input)

    def get_params(self, deep=True):
        params = dict(x=self.x, nb_epoch=self.nb_epoch, learning_rate=self.learning_rate, layers=self.layers,
                      batch_size=self.batch_size, neurons=self.neurons, activation=self.activation,
                      output_activation=self.output_activation)
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, x):
        """
        Outputs the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        X, _ = self._preprocessor(x, training=False)
        y_tensor = self.model.forward(X)
        return y_tensor.detach().numpy()

    def score(self, x, y):
        """
        Evaluates the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        X, Y = self._preprocessor(x, y=y, training=False)
        pred = self.predict(x)

        return np.sqrt(mean_squared_error(pred, y))

    def batch_loader(self, x, y, shuffle=True):
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return loader


class ReplayBuffer:
    def __init__(self, size):
        """Replay buffer initialisation

        Args:
            size(int): capacity of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)

    def push(self, transition):
        """Push an object to the replay buffer

        Args:
            transition(obj): to be stored in replay buffer

        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size):
        """Randomly sample the replay buffer

        Args:
            batch_size: size of sample

        Returns:
            sampled list from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)


class Agent:
    def __init__(self, env):
        """Object for a DQN based agent

        Args:
            env (object): 
        """
        self.training_dict = None
        self.policy_dqn = None
        self.env = env
        self.target_dqn = None
        self.label = 'DQN Agent'
        self.threshold = self.env.spec.reward_threshold

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
                self.training_dict(dict): Dictionary with some episode rewards and durations
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
                print("episode ", i_episode + 1, "of", episodes)

            observation, info = self.env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            episode_reward = 0

            if epsilon > epsilon_end:
                epsilon *= eps_decay
                epsilon = max(epsilon_end, epsilon)

            while not (done or terminated):

                # Select and perform an action
                action = epsilon_greedy(epsilon, self.policy_dqn, state)

                observation, reward, done, terminated, info = self.env.step(action)
                episode_reward += reward
                if clip_rewards:
                    reward = clip_reward(reward)

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

        print("Training is complete")
        if show_time:
            endt = time.time()
            self.print_time(strt, endt)

        # print(self.policy_dqn.state_dict())
        self.training_dict = dict(episode_durations=episode_durations, episode_rewards=episode_rewards)
        return self.training_dict

    def evaluate_agent(self, episodes, plots=True):
        """Evaluates performance of Trained Agent over a number of episodes
            Returns:
                Greedy action according to DQN
            """

        episode_durations = []
        episode_rewards = []
        # print(self.policy_dqn.state_dict())
        print("Evaluating Trained Agent...")

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

            while not (done or terminated):
                # frames.append(self.env.render(render_mode='rgb_array'))
                frames.append(self.env.render())

                # Select and perform an action
                action = greedy_action(self.policy_dqn, state)

                observation, reward, done, terminated, info = self.env.step(action)
                episode_reward += reward

                if done or terminated:
                    episode_durations.append(t + 1)
                    episode_rewards.append(episode_reward)
                    # Save the frames as a gif
                    imageio.mimsave('images/evaluations_' + str(i_episode+1) + '.gif', frames, fps=5)

                t += 1
        if plots:
            self.plot_episodes(episode_rewards)

        return dict(episode_durations=episode_durations, episode_rewards=episode_rewards)

    def update_target(self):
        """Update target network parameters using policy network.

        Args:
            target_dqn: target network to be modified in-place
            policy_dqn: the DQN that selects the action
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

    @staticmethod
    def epsilon_greedy(epsilon, dqn, state):
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
            return greedy_action(dqn, state)
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
            Q_vals = self.target_dqn(next_states).gather(1, policy_dqn_actions).reshape(-1)
            bellman_targets = (~dones).reshape(-1) * Q_vals + rewards.reshape(-1)
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


def train_agent(env, num_runs, dqn_params, replay_buffer, episodes, epsilon, epsilon_end=0.01, eps_decay=1,
                update_frequency=10, batch_size=32, clip_rewards=False):
    """Trains the dqn
        Returns:
            Greedy action according to DQN
        """
    runs_results = []
    runs_rewards = []

    for run in range(num_runs):
        print(f"Starting run {run + 1} of {num_runs}")
        policy_dqn = DQN(**dqn_params)
        target_dqn = DQN(**dqn_params)
        update_target(target_dqn, policy_dqn)
        target_dqn.eval()

        optimizer = policy_dqn.optim
        memory = ReplayBuffer(replay_buffer)

        episode_durations = []
        episode_rewards = []

        for i_episode in range(episodes):
            if (i_episode + 1) % (episodes / 10) == 0:
                print("episode ", i_episode + 1, "/", episodes)

            observation, info = env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            episode_reward = 0

            if epsilon > epsilon_end:
                epsilon *= eps_decay
                epsilon = max(epsilon_end, epsilon)

            while not (done or terminated):

                # Select and perform an action
                action = epsilon_greedy(epsilon, policy_dqn, state)

                observation, reward, done, terminated, info = env.step(action)
                episode_reward += reward
                if clip_rewards:
                    reward = clip_reward(reward)

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
                    mse_loss = loss(policy_dqn, target_dqn, state_batch, action_batch, reward_batch, nextstate_batch,
                                    dones)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()

                if done or terminated:
                    episode_durations.append(t + 1)
                    episode_rewards.append(episode_reward)
                t += 1

            # Update the target dqn, copying all weights and biases in DQN
            if i_episode % update_frequency == 0:
                update_target(target_dqn, policy_dqn)
        runs_results.append(episode_durations)
        runs_rewards.append(episode_rewards)
    print("Training is complete")
    return dict(policy_dqn=policy_dqn, target_dqn=target_dqn, runs_results=runs_results, runs_rewards=runs_rewards)


def update_target(target_dqn, policy_dqn):
    """Update target network parameters using policy network.

    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())


def greedy_action(dqn, state):
    """Select action according to a given DQN

    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))


def epsilon_greedy(epsilon, dqn, state):
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
        return greedy_action(dqn, state)
    else:
        return random.randint(0, num_actions - 1)


def loss(policy_dqn, target_dqn, states, actions, rewards, next_states, dones, is_DQN=True):
    """Calculate Bellman error loss

    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
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
        bellman_targets = (~dones).reshape(-1) * (target_dqn(next_states)).max(1).values + rewards.reshape(-1)
        q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    elif not is_DQN:
        # The above code first determines the ideal actions using the policy network,
        # and then computes their Q Values using the target network
        policy_dqn_actions = policy_dqn(next_states).max(1).indices.reshape([-1, 1])
        Q_vals = target_dqn(next_states).gather(1, policy_dqn_actions).reshape(-1)
        bellman_targets = (~dones).reshape(-1) * Q_vals + rewards.reshape(-1)
        q_values = policy_dqn(states).gather(1, actions).reshape(-1)

    return ((q_values - bellman_targets) ** 2).mean()


def clip_reward(reward, a=-1, b=1):
    if reward < a:
        return a
    elif reward > b:
        return b
    else:
        return reward


if __name__ == '__main__':
    # env = gym.make("LunarLander-v2", render_mode="human")
    env = gym.make("LunarLander-v2", render_mode='rgb_array')

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    dqn_agent = Agent(env)

    # DQN Parameters
    layers = [input_size, 64, 64, output_size]  # DQN Architecture
    activation = 'relu'
    weights = 'xnorm'
    optim = 'Adam'
    learning_rate = 5e-4
    dqn_params = dict(layers=layers, activation=activation, weights=weights, optim=optim, learning_rate=learning_rate)

    # Training Parameters
    num_runs = 1
    epsilon = 1
    eps_decay = 0.995  # Epsilon is reduced by 1-eps_decay every episode
    replay_buffer = 100000
    batch_size = 128
    epsilon_end = 0.01
    episodes = 1000
    update_frequency = 5
    clip_rewards = False

    training_params = dict(epsilon=epsilon, eps_decay=eps_decay, replay_buffer=replay_buffer,
                           batch_size=batch_size, epsilon_end=epsilon_end, episodes=episodes,
                           update_frequency=update_frequency, dqn_params=dqn_params, clip_rewards=clip_rewards)

    run_stats = dqn_agent.train_agent(show_time=True, **training_params)
    dqn_agent.plot_episodes(run_stats['episode_rewards'])
    dqn_agent.evaluate_agent(10, plots=True)

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
