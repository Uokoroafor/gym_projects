# OpenAI Gym Projects:
## DQN Agents for Reinforcement Learning

This repository contains an implementation of a Deep Q-Network (DQN) agent for reinforcement learning tasks, specifically designed to interact with OpenAI Gym environments, particularly the classic control environments with discrete action spaces. 
The agent is implemented in PyTorch and uses a simple feedforward network as the Q-function approximator. 
The agent is capable of training and evaluating its performance in a variety of environments.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Uokoroafor/gym_projects.git
```

2. Install the required dependencies:

```bash
cd gym_projects
pip install -r requirements.txt
```

## Usage

The repository provides examples of using the DQN agent to solve the [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) and [Cartpole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environments. You can find the examples in the `examples/` folder.

To train the DQN agent on the  LunarLander environment, for example, run the following command:

```bash
python examples/lunar_agents.py
```

The agent is able to learn the environment going from a random policy (left) to a policy that is able to land the Lander on the landing pad (right).

<p align="center">
  <img src="images/lunarlander-v2/random.gif" alt="Random Policy" width="300"/>
  <img src="images/lunarlander-v2/evaluation.gif" alt="Trained Policy" width="300"/>


```bash
python examples/cartpole_agents.py
```

The agent is able to the LunarLander V2 (environment details can be found [here](https://gymnasium.farama.org/environments/box2d/lunar_lander/)) going from a random policy to a policy that is able to land the LunarLander on the landing pad -see the GIFs below.

<p align="center">
  <img src="images/cartpole-v1/random.gif" alt="Random Policy" width="300"/>
  <img src="images/cartpole-v1/evaluations.gif" alt="Trained Policy" width="300"/>

Please note that the agent doesn't work with environments that have continuous action spaces. It also doesn't work on the MountainCar-v0 environment, as the agent is not able to learn a good policy for these environments.
## Project Structure

The repository has the following structure:

```
├── dqn.py                 # DQN model implementation
├── replaybuffer.py        # Replay buffer implementation
├── agent.py               # DQN agent implementation
├── examples/              # Example usage of the DQN agent
├── images/                # Directory for saving rendering images and GIFs
└── README.md              # Project README file
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or a pull request.

## Acknowledgments

This project builds on a Reinforcement Learning coursework from Imperial College London and was also inspired by the Deep Q-Network algorithm and the OpenAI Gym environment.

## References
1. **Playing Atari with Deep Reinforcement Learning** by Volodymyr Mnih et al. (2013):
   - [ArXiv Preprint](https://arxiv.org/abs/1312.5602)

2. **Human-level control through deep reinforcement learning** by Volodymyr Mnih et al. (2015):
   - [Nature Journal](https://www.nature.com/articles/nature14236)
   - [ArXiv Preprint](https://arxiv.org/abs/1509.06461)

- [OpenAI Gym.](https://gym.openai.com/)



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)