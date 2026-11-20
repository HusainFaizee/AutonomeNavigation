import numpy as np
import config

from main import run_experiment
from utils.plot import plot_multiple


all_results = {}

alphas = [0.1, 0.2]
gammas = [0.9, 0.99]
epsilons = [0.995, 0.99]

for alpha in alphas:
    for gamma in gammas:
        for epsilon in epsilons:

            config.ALPHA = alpha
            config.GAMMA = gamma
            config.EPSILON_DECAY = epsilon

            rewards = run_experiment("q_learning")

            label = (
                f"a={alpha}, g={gamma}, e={epsilon}"
            )

            all_results[label] = rewards

            score = np.mean(rewards[-50:])

            print(
                f"alpha={alpha}, "
                f"gamma={gamma}, "
                f"epsilon={epsilon}, "
                f"score={score}"
            )
            

# ONE FINAL GRAPH
plot_multiple(all_results)
algorithms = [
    "q_learning",
    "sarsa",
    "dqn"
]

for algo in algorithms:

    rewards = run_experiment(algo)

    all_results[algo] = rewards