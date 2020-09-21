from tqdm import tqdm


def runner(agent, agent_normalized, user_contexts, time_steps):
    # main loop 
    regrets = []
    for t in tqdm(range(time_steps)):
        # sample user
        user = user_contexts.sample_user()
        # sample contexts
        contexts = user_contexts.sample_contexts(user)
        # choose context
        chosen_context = agent.choose(user, contexts, t)
        # sample reward and normalize reward with random choice
        reward = user_contexts.sample_reward(user, chosen_context) - \
                user_contexts.sample_reward(user, agent_normalized.choose(user, contexts, t))
        # update agent
        agent.update(user, chosen_context, reward)
        # collect regrets
        if t != 0: regrets.append(regrets[t-1] + reward)
        else: regrets.append(reward)
    return regrets