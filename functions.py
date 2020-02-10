def play_one_episode(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        env.render()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        agent.train(state, next_state, reward, done)
        state = next_state.copy()
        return total_reward
