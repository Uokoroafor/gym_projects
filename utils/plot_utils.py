import matplotlib.pyplot as plt
import torch
from typing import List, Optional


def plot_episodes(episode_stats: List[float], title: Optional[str] = None, threshold: Optional[float] = None):
    """Plot the scores of the episodes.

    Args:
        episode_stats: List of scores for each episode.
        title: Title of the plot (if any).
        threshold: Threshold for the plot (if any).
    """
    rewards = torch.tensor(episode_stats)
    if rewards.dim() == 2:
        means = rewards.float().mean(1)
        stds = rewards.float().std(1)
    else:
        means = rewards.float()
        stds = rewards.float().std(0)

    episodes = len(means)

    plt.plot(torch.arange(1, episodes + 1), means, color="g")
    plt.ylabel("score")
    plt.xlabel("episode")
    if rewards.dim() == 2:
        plt.fill_between(torch.arange(1, episodes + 1), means, means + stds, alpha=0.3, color='g')
        plt.fill_between(torch.arange(1, episodes + 1), means, means - stds, alpha=0.3, color='g')
    if threshold is not None:
        plt.axhline(
            y=threshold, color="r", linestyle="dashed", label="Solved Threshold"
        )
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()
