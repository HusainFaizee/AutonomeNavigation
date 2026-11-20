import time
def train_sarsa(env, agent, config, render = False):
    rewards = []

    for ep in range(config.EPISODES):
        state = env.reset()
        action = agent.choose_action(state)

        total_reward = 0
        done = False

        while not done:
            if render and ep == config.EPISODES -1:
                env.render(
                   episode=ep,
                   step=env.steps,
                   agent=agent
                )
                time.sleep(0.1)
            next_state, reward, done = env.step(action)

            next_action = agent.choose_action(next_state)

            agent.update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action
            total_reward += reward

        # epsilon decay
        agent.epsilon = max(config.EPSILON_MIN,
                    agent.epsilon * config.EPSILON_DECAY)

        rewards.append(total_reward)

    return rewards