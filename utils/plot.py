import matplotlib.pyplot as plt
import numpy as np


def plot_multiple(results):
    plt.figure(figsize=(12, 6))

    for label, rewards in results.items():

        rewards = np.array(rewards)

        # moving average
        window = 20

        if len(rewards) >= window:
            smoothed = np.convolve(
                rewards,
                np.ones(window) / window,
                mode="valid"
            )

            plt.plot(smoothed, label=label)

        else:
            plt.plot(rewards, label=label)

    plt.title("RL Algorithm Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    plt.show()