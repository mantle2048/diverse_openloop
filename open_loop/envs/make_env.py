from typing import Dict
import gym
import dmc2gym
from gym.wrappers import ClipAction

from open_loop.envs.wrappers.trajectory_generator_wrapper_env import TrajectoryGeneratorWrapperEnv
from open_loop.trajectory_generator import CpgRbfNet

TIMESTEP = {
    "MinitaurBulletEnv-v0": 0.002,
    "HalfCheetah-v3": 0.05,
    "Ant-v3": 0.05,
    "Swimmer-v3": 0.04,
    "AntBulletEnv-v0": 0.0165,
    "HalfCheetahBulletEnv-v0": 0.0165,
    "HopperBulletEnv-v0": 0.0165,
    "cheetah_run": 0.01,
    "MinitaurTrottingEnv-v0": 0.01,
    "MinitaurReactiveEnv-v0": 0.01,
}

def make_env(env_name, seed=0):

    env_type='dmc' if '_' in env_name else 'gym'

    if env_type == 'gym':
        if 'Env' in env_name:
            # env = gym.make(env_name, disable_env_checker=True)
            env = gym.make(env_name)
            env.seed(seed)
            env.action_space.seed(seed)
        else:
            env = gym.make(env_name)
            env.seed(seed)
            env.action_space.seed(seed)
    elif env_type == 'dmc':
        domain, task = tuple(env_name.split('_'))
        env = dmc2gym.make(domain_name=domain, task_name=task, seed=seed)
    else:
        raise ValueError("No supported env type, avaiable env_tpye = [gym, dmc]")
    env.dt = TIMESTEP[env_name]
    env = ClipAction(env)
    return env

def wrap_env(env, config: Dict):

    sin_config = {
        'amplitude': config['amplitude'],
        'theta': config['theta'],
        'frequency': config['frequency'],
    }
    trajectory_generator = CpgRbfNet(
        sin_config, config['timestep'], config['num_rbf'], config['num_act']
    )

    env = TrajectoryGeneratorWrapperEnv(
        env, trajectory_generator
    )
    return env

