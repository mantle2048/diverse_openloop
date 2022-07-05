import gym

def make_env(env_name, seed=0):
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
