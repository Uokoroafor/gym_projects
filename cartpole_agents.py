from agent_utils import Agent
import gym


if __name__ == '__main__':

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    dqn_agent = Agent(env)

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # DQN Parameters
    layers = [input_size, 32, 32, output_size]  # DQN Architecture
    activation = 'relu'
    weights = 'xunif'
    optim = 'Adam'
    learning_rate = 1e-3
    dqn_params = dict(layers=layers, activation=activation, weights=weights, optim=optim, learning_rate=learning_rate)

    # Training Parameters
    epsilon = 1
    eps_decay = 0.995  # Epsilon is reduced by 1-eps_decay every episode
    replay_buffer = 100000
    batch_size = 64
    epsilon_end = 0.01
    episodes = 1000
    update_frequency = 5
    clip_rewards = False

    training_params = dict(epsilon=epsilon, eps_decay=eps_decay, replay_buffer=replay_buffer,
                           batch_size=batch_size, epsilon_end=epsilon_end, episodes=episodes,
                           update_frequency=update_frequency, dqn_params=dqn_params, clip_rewards=clip_rewards)

    run_stats = dqn_agent.train_agent(show_time=True, **training_params)
    dqn_agent.plot_episodes(run_stats['episode_rewards'])
    dqn_agent.evaluate_agent(10, plots=True, save_every=10)


