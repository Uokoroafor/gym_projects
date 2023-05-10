import time
from collections import deque
import gym
import torch
from dqn import DQN
from replaybuffer import ReplayBuffer
from agent_utils import Agent


class MountainCarAgent(Agent):
    def __init__(self, env):
        super().__init__(env)

    # Overwrite the train_agent method to maximise the product of the distance travelled and the speed
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

        print(f'{self.threshold} is the threshold for {self.label}.')

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
                action = self.epsilon_greedy(epsilon, self.policy_dqn, state)

                observation, reward, done, terminated, _ = self.env.step(action)
                # episode_reward += reward

                # Overwrite the reward function to maximise the product of the distance travelled and the speed
                episode_reward += observation[0] * observation[1]

                # print(f'Episode reward: {episode_reward}')
                # print(f'Distance: {observation[0]}')
                # print(f'Speed: {observation[1]}')

                # want to scale so that velocity is more important than distance

                reward = observation[0] * observation[1] + observation[0] * 0.1

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

            # # Check if solved threshold has been reached
            # if np.mean(scores_window) >= self.threshold:
            #     print(f'Environment solved within {i_episode + 1} episodes.')
            #     print(f'Average Score: {np.mean(scores_window)}')
            #     break

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


if __name__ == '__main__':
    env = gym.make("MountainCar-v0", render_mode='rgb_array')

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    dqn_agent = MountainCarAgent(env)

    # DQN Parameters
    layers = [input_size, 32, 32, output_size]
    activation = 'relu'
    weights = 'xunif'
    optim = 'Adam'
    learning_rate = 1e-4
    dqn_params = dict(layers=layers, activation=activation, weights=weights, optim=optim, learning_rate=learning_rate)

    # Training Parameters
    epsilon = 1
    eps_decay = 0.996  # Epsilon is reduced by 1-eps_decay every episode
    replay_buffer = 100000
    batch_size = 64
    epsilon_end = 0.01
    episodes = 1000
    update_frequency = 5
    clip_rewards = False
    delay_decay = False
    gamma = 0.9

    training_params = dict(epsilon=epsilon, eps_decay=eps_decay, replay_buffer=replay_buffer,
                           batch_size=batch_size, epsilon_end=epsilon_end, episodes=episodes,
                           update_frequency=update_frequency, dqn_params=dqn_params, clip_rewards=clip_rewards,
                           delay_decay=delay_decay, gamma=gamma)

    run_stats = dqn_agent.train_agent(show_time=True, **training_params)
    dqn_agent.plot_episodes(run_stats['episode_rewards'])
    dqn_agent.evaluate_agent(10, plots=True, save_every=10)
