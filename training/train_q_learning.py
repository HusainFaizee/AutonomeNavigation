import time
import numpy as np


def train_q_learning(env, agent, config, render=False):

    rewards = []

    for ep in range(config.EPISODES):

        state = env.reset()

        total_reward = 0
        done = False

        while not done:

            if render and ep == config.EPISODES - 1:

                env.render(
                    episode=ep,
                    step=env.steps,
                    agent=agent
                )

                time.sleep(0.1)

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state

            total_reward += reward

        # FIXED epsilon decay
        agent.epsilon = max(
            config.EPSILON_MIN,
            agent.epsilon * config.EPSILON_DECAY
        )

        rewards.append(total_reward)

        # progress logging
        if ep % 50 == 0:

            avg_last = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)

            print(
                f"Episode {ep:3d} | "
                f"Reward = {total_reward:6.2f} | "
                f"Avg(50) = {avg_last:6.2f} | "
                f"Epsilon = {agent.epsilon:.3f}"
            )

    return rewards