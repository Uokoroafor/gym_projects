import random
import time
from collections import deque
from typing import Optional, Dict

import gym
import torch
from torch import nn
from dqn import DQN
from replaybuffer import ReplayBuffer
from utils.logging_utils import DQNLogger, get_time
from utils.plot_utils import plot_episodes
from utils.render_utils import save_render


class Agent:
    def __init__(self, env: gym.Env, reward_threshold: Optional[float] = None):
        """Creates a DQN based agent for a given gym environment

        Args:
            env (gym.Env): The environment to be solved
            reward_threshold (float): The reward threshold for the environment. If None, the default threshold
            for the environment is used


        """
        self.trained_state_dict = None
        self.training_dict = None
        self.policy_dqn = None
        self.env = env
        self.target_dqn = None
        self.label = "DQN Agent"
        if reward_threshold is None:
            self.threshold = self.env.spec.reward_threshold
        else:
            self.threshold = reward_threshold
        self.logger = None

    def train_agent(
            self,
            dqn_params: Dict,
            replay_buffer: ReplayBuffer,
            episodes: int,
            epsilon: float,
            epsilon_end: Optional[float] = 0.01,
            eps_decay: Optional[float] = 1.0,
            gamma: Optional[float] = 1.0,
            update_frequency: Optional[int] = 10,
            batch_size: Optional[int] = 32,
            clip_rewards: Optional[bool] = False,
            show_time: Optional[bool] = False,
            delay_decay: Optional[bool] = False,
            log_path: Optional[str] = None,
            verbose: Optional[bool] = False,
    ):
        """Trains the Agent using the specified parameters
        Args:
            dqn_params (Dict): Parameters for the DQN
            replay_buffer (ReplayBuffer): Replay buffer to be used
            episodes (int): Number of episodes to train for
            epsilon (float): Starting epsilon value
            epsilon_end (Optional[float], optional): End epsilon value. Defaults to 0.01.
            eps_decay (Optional[float], optional): Decay rate of epsilon. Defaults to 0.995.
            gamma (Optional[float], optional): Discount factor. Defaults to 1.0.
            update_frequency (Optional[int], optional): Frequency of updating the target network. Defaults to 10.
            batch_size (Optional[int], optional): Batch size for training. Defaults to 32.
            clip_rewards (Optional[bool], optional): Whether to clip rewards. Defaults to False.
            show_time (Optional[bool], optional): Whether to show time taken for training. Defaults to False.
            delay_decay (Optional[bool], optional): Whether to delay the decay of epsilon. Defaults to False.
            log_path (Optional[str], optional): Path to save logs. Defaults to None.
            verbose (Optional[bool], optional): Whether to print training logs. Defaults to False.
        """

        assert (
                epsilon >= epsilon_end
        ), "Starting epsilon should not be less that end epsilon"

        strt = time.time()

        # Initialize the policy and target networks
        self.policy_dqn = DQN(**dqn_params)
        self.target_dqn = DQN(**dqn_params)
        self.update_target()
        self.target_dqn.eval()

        # Initialize the optimizer and replay buffer
        optimizer = self.policy_dqn.optim
        memory = ReplayBuffer(replay_buffer)

        # Initialize the logger
        episode_durations = []
        episode_rewards = []
        scores_window = deque(maxlen=100)

        # Initialize the logger
        log_path = 'training_logs/dqn_log_' + time.strftime("%y%m%d_%H%M%S") + '.txt' if log_path is None else log_path
        logger = DQNLogger(log_path, self.label, verbose=verbose)
        if self.logger is None:
            self.logger = logger
        logger.log_info(f"Training {self.label} for {self.env.unwrapped.spec.id}...")

        for i_episode in range(episodes):
            if (i_episode + 1) % (episodes / 10) == 0:
                logger.log_info(f"episode {i_episode + 1} of max {episodes}")

            observation, _ = self.env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            episode_reward = 0

            while not (done or terminated):
                # Select and perform an action
                action = self.epsilon_greedy(epsilon, self.policy_dqn, state)

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
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (
                        torch.stack(x) for x in zip(*transitions)
                    )
                    # Compute loss
                    mse_loss = self.loss(
                        state_batch,
                        action_batch,
                        reward_batch,
                        nextstate_batch,
                        dones,
                        gamma,
                    )
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()

                if done or terminated:
                    episode_durations.append(t + 1)
                    episode_rewards.append(episode_reward)
                    scores_window.append(episode_reward)

                    if delay_decay and episode_reward > 0:
                        delay_decay = False
                        logger.log_info(
                            f"Received positive reward at episode {i_episode}.Will begin epsilon decay now")

                t += 1

            # Update the target dqn, copying all weights and biases in DQN
            if i_episode % update_frequency == 0:
                self.update_target()

            # Check if solved threshold has been reached
            # Solved threshold is the average of the last 100 episodes rewards and that the last episode is a success
            if torch.mean(torch.tensor(scores_window)) >= self.threshold and episode_reward >= self.threshold:
                logger.log_info(f"Environment solved within {i_episode + 1} episodes.")
                logger.log_info(f"Average Score: {torch.mean(torch.tensor(scores_window)): .2f}")
                break

            # Update epsilon
            if epsilon > epsilon_end and not delay_decay:
                epsilon *= eps_decay
                epsilon = max(epsilon_end, epsilon)

        logger.log_info("Training is complete")
        endt = time.time()
        logger.log_info(get_time(strt, endt, "Total Training Time: "))

        self.training_dict = dict(
            episode_durations=episode_durations, episode_rewards=episode_rewards
        )
        return self.training_dict

    def evaluate_agent(self, episodes: int, plots: Optional[bool]=True, save_every: Optional[int]=None, nb_render: Optional[bool]=False):
        """Evaluates performance of Trained Agent over a number of episodes
        Args:
            episodes (int): Number of episodes to evaluate the agent
            plots (Optional[bool], optional): Whether to plot the rewards and durations. Defaults to True.
            save_every (Optional[int], optional): Save the agent every save_every episodes. Defaults to None.
            nb_render (Optional[bool], optional): Whether to render the environment. Defaults to False.

        Returns:
            dict: Dictionary with episode rewards and durations
        """

        episode_durations = []
        episode_rewards = []

        # Set save_every so that it is not a divisor for any number in range(episodes)
        if save_every is None:
            save_every = episodes + 2

        # Initialize the logger if it is not already initialized
        if self.logger is None:
            self.logger = DQNLogger(None, self.label, verbose=False)

        self.logger.log_info("Evaluating Trained Agent...")

        # Turn off train mode
        self.policy_dqn.eval()

        for i_episode in range(episodes):
            if (i_episode + 1) % (episodes / 10) == 0:
                self.logger.log_info(f"Evaluation Episode {i_episode + 1} of {episodes}")

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
                    self.logger.log_info(f"Episode {i_episode + 1} with reward {episode_reward:.2f}")
                    self.logger.log_info(f"{t + 1} steps")

                    if ((i_episode + 1) % save_every) == 0:
                        save_render(self.env, frames, i_episode, nb_render=nb_render)

                t += 1
        if plots:
            plot_episodes(episode_rewards, title=f"Evaluation of {self.label} for {self.env.unwrapped.spec.id}",
                          threshold=self.threshold)

        return dict(
            episode_durations=episode_durations, episode_rewards=episode_rewards
        )

    def update_target(self):
        """Update target network parameters using policy network.

        Returns:
            None
        """

        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

    @staticmethod
    def greedy_action(dqn: nn.Module, state: torch.Tensor) -> int:
        """Select action according to a given DQN

        Args:
            dqn (nn.Module): the DQN that estimates the action values
            state (torch.Tensor): state at which the action is chosen

        Returns:
            int: greedy action
        """
        return int(torch.argmax(dqn(state)))

    def epsilon_greedy(self, epsilon: float, dqn: nn.Module, state: torch.Tensor) -> int:
        """Sample an epsilon-greedy action according to a given DQN

        Args:
            epsilon (float): epsilon value
            dqn (nn.Module): the DQN that estimates the action values
            state (torch.Tensor): state at which the action is chosen

        Returns:
            int: epsilon-greedy action
        """
        q_values = dqn(state)
        num_actions = q_values.shape[0]
        p = random.random()
        if p > epsilon:
            return self.greedy_action(dqn, state)
        else:
            return random.randint(0, num_actions - 1)

    def loss(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, gamma: float) -> float:
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

        bellman_targets = gamma * (~dones).reshape(-1) * (
            self.target_dqn(next_states)
        ).max(1).values + rewards.reshape(-1)
        q_values = self.policy_dqn(states).gather(1, actions).reshape(-1)

        return ((q_values - bellman_targets) ** 2).mean()

    @staticmethod
    def clip_reward(reward: float, a: float, b: float) -> float:
        """Clip reward to be in the range [a, b]

        Args:
            reward (float): reward to be clipped
            a (float): lower bound
            b (float): upper bound

        Returns:
            float: clipped reward
        """
        if reward < a:
            return a
        elif reward > b:
            return b
        else:
            return reward


class DDQNAgent(Agent):
    def __init__(self, env: gym.Env, reward_threshold: Optional[float] = None):
        """Initialize the DDQN Agent

        Args:
            env (gym.env): Gym environment
            reward_threshold (float, optional): Reward threshold for the environment. Defaults to None.
        """

        super().__init__(env=env, reward_threshold=reward_threshold)
        self.label = "DDQN Agent"

    def loss(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, gamma: float) -> float:
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

        policy_dqn_actions = (
            self.policy_dqn(next_states).max(1).indices.reshape([-1, 1])
        )
        q_vals = self.target_dqn(next_states).gather(1, policy_dqn_actions).reshape(-1)
        bellman_targets = gamma * (~dones).reshape(-1) * q_vals + rewards.reshape(-1)
        q_values = self.policy_dqn(states).gather(1, actions).reshape(-1)

        return ((q_values - bellman_targets) ** 2).mean()


def dqn_example(gym_env: gym.Env):
    dqn_agent = Agent(gym_env)

    input_size = gym_env.observation_space.shape[0]
    output_size = gym_env.action_space.n

    # DQN Parameters
    layers = [input_size, 256, 128, output_size]
    activation = "relu"
    weights = "xunif"
    optim = "Adam"
    learning_rate = 5e-4
    dqn_params = dict(
        layers=layers,
        activation=activation,
        weights=weights,
        optim=optim,
        learning_rate=learning_rate,
    )

    # Training Parameters
    epsilon = 1
    eps_decay = 0.995
    replay_buffer = 100000
    batch_size = 128
    epsilon_end = 0.01
    episodes = 100
    update_frequency = 5
    clip_rewards = False

    training_params = dict(
        epsilon=epsilon,
        eps_decay=eps_decay,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        epsilon_end=epsilon_end,
        episodes=episodes,
        update_frequency=update_frequency,
        dqn_params=dqn_params,
        clip_rewards=clip_rewards,
    )

    run_stats = dqn_agent.train_agent(show_time=True, **training_params)
    plot_episodes(run_stats["episode_rewards"], "DQN Agent", dqn_agent.threshold)


def ddqn_example(gym_env: gym.Env):
    """Example of how to use the DDQN Agent to solve an environment"""
    ddqn_agent = DDQNAgent(gym_env)

    input_size = gym_env.observation_space.shape[0]
    output_size = gym_env.action_space.n

    # DDQN Parameters
    layers = [input_size, 256, 128, output_size]  # DDQN Architecture
    activation = "relu"
    weights = "xunif"
    optim = "Adam"
    learning_rate = 5e-4
    dqn_params = dict(
        layers=layers,
        activation=activation,
        weights=weights,
        optim=optim,
        learning_rate=learning_rate,
    )

    # Training Parameters
    epsilon = 1
    eps_decay = 0.995
    replay_buffer = 100000
    batch_size = 64
    epsilon_end = 0.01
    episodes = 1000
    update_frequency = 5
    clip_rewards = False

    training_params = dict(
        epsilon=epsilon,
        eps_decay=eps_decay,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        epsilon_end=epsilon_end,
        episodes=episodes,
        update_frequency=update_frequency,
        dqn_params=dqn_params,
        clip_rewards=clip_rewards,
    )

    run_stats = ddqn_agent.train_agent(show_time=True, **training_params)
    plot_episodes(run_stats["episode_rewards"], "DDQN Agent", ddqn_agent.threshold)


if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    dqn_example(gym_env=env)
