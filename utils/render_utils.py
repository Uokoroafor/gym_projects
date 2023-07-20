from typing import Optional, List

import gym
from gym import Env
import imageio
from imageio.core.util import Image
from utils.file_utils import check_path_exists
import time


def save_random_renders(gym_env: Env, episodes: Optional[int] = 10, save_every: Optional[int] = 10,
                        nb_render: Optional[bool] = 1):
    """Evaluates performance of Trained Agent over a number of episodes
    Args:
        gym_env: Gym environment
        episodes(int): Number of episodes the train agent carries out
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

        gym_env.reset()

        done = False
        terminated = False
        t = 0
        episode_reward = 0
        frames = []

        while not (done or terminated):
            frames.append(gym_env.render())

            # Select and perform a random action
            action = gym_env.action_space.sample()

            _, reward, done, terminated, _ = gym_env.step(action)
            episode_reward += reward

            if done or terminated:
                episode_durations.append(t + 1)
                episode_rewards.append(episode_reward)
                print(
                    f"Random episode {i_episode + 1} with reward {episode_reward}"
                )
                print(f"{t + 1} steps")

                if ((i_episode + 1) % save_every) == 0:
                    save_render(gym_env,
                                frames, i_episode, mode="rand", nb_render=nb_render
                                )

            t += 1

    return episode_rewards


def save_render(gym_env: Env, frames: List[Image], i_episode: int, mode: Optional[str] = "eval",
                nb_render: Optional[bool] = False):
    """
    Saves the rendering as a gif
    Args:
        gym_env (gym.Env): Gym environment
        frames (list(images)): list of image frames saved from rendering
        i_episode (int): the current episode of learning
        mode (str): where in 'eval' (evaluating trained agent), 'random'(evaluating untrained agent) or 'training' (saving renderings of an agent in training) mode
        nb_render (bool): if True, indicates that the rendering is for an ipynb and is saved without a timestamp.

    Returns:
        None

    """
    folder_name = "../images/" + gym_env.unwrapped.spec.id
    check_path_exists(folder_name)

    if mode == "eval":
        mode = "evaluation_"
    elif mode == "rand":
        mode = "random_"
    else:
        mode = "training_"

    if nb_render:
        imageio.mimsave(str(folder_name) + "/" + mode[:-1] + ".gif", frames, fps=15)

    else:
        imageio.mimsave(
            str(folder_name)
            + "/"
            + mode
            + str(i_episode + 1)
            + "_"
            + time.strftime("%y%m%d_%H%M")
            + ".gif",
            frames,
            fps=15,
        )
