# DQN Agent for Reinforcement Learning

This repository contains an implementation of a Deep Q-Network (DQN) agent for reinforcement learning tasks, specifically designed to interact with OpenAI Gym environments. The agent is capable of training and evaluating its performance in a variety of environments.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/dqn-agent.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install Gym environments:

```bash
pip install gym
```

## Usage

The repository provides examples of using the DQN agent to solve the LunarLander-v2 environment. You can find the examples in the `examples.py` file.

To train the DQN agent on the LunarLander-v2 environment, run the following command:

```bash
python examples.py
```

To use the Double DQN variant, you can uncomment the corresponding lines in the `examples.py` file.

## Project Structure

The repository has the following structure:

```
├── dqn.py                 # DQN model implementation
├── replaybuffer.py        # Replay buffer implementation
├── agent.py               # DQN agent implementation
├── examples.py            # Example usage of the DQN agent
├── images/                # Directory for saving rendering images and GIFs
└── README.md              # Project README file
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or a pull request.

## Acknowledgments

This project was inspired by the Deep Q-Network algorithm and the OpenAI Gym environment.

## References
- [OpenAI Gym.](https://gym.openai.com/)
