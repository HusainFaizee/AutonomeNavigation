import numpy as np

print("MAIN STARTED")

from env.gridworld import GridWorld

from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn import DQNAgent

from training.train_q_learning import train_q_learning
from training.train_sarsa import train_sarsa
from training.train_dqn import train_dqn

from utils.plot import plot_multiple


import config


AGENTS = {
    "q_learning": (QLearningAgent, train_q_learning),
    "sarsa": (SARSAAgent, train_sarsa),
    "dqn": (DQNAgent, train_dqn),
}


def run_experiment(agent_name="sarsa"):

    print("RUN EXPERIMENT STARTED")

    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}")

    AgentClass, train_fn = AGENTS[agent_name]

    all_runs = []

    for run in range(config.RUNS):

        print(f"\n========== RUN {run + 1}/{config.RUNS} ==========\n")

        env = GridWorld(
            size=config.GRID_SIZE,
            start=(0, 0),
            goal=config.GOAL,
            obstacles=config.OBSTACLES,
            max_steps=config.MAX_STEPS
        )

        agent = AgentClass(
            env=env,
            alpha=config.ALPHA,
            gamma=config.GAMMA,
            epsilon=config.EPSILON_START
        )

        print(
            f"Training with:\n"
            f"Alpha   = {config.ALPHA}\n"
            f"Gamma   = {config.GAMMA}\n"
            f"Epsilon = {config.EPSILON_START}\n"
        )

        rewards = train_fn(
            env,
            agent,
            config,
            render=True
        )

        print(f"Run {run + 1} completed.")

        all_runs.append(rewards)

    avg_rewards = np.mean(all_runs, axis=0)

    print("\nTRAINING FINISHED\n")

    print("Final Average Reward:")
    print(np.mean(avg_rewards[-50:]))

    return avg_rewards


if __name__ == "__main__":

    rewards = run_experiment(agent_name="dqn")

    plot_multiple({
        "DQN": rewards
    })

    print("\nPROGRAM COMPLETE\n")