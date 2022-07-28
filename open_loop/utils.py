import os
import os.path as osp
import torch
import json
import gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
from torch import nn
from open_loop.user_config import LOCAL_LOG_DIR
from open_loop.envs.make_env import make_env
from open_loop.rollout import rollout, rollouts, traj_rollouts, traj_rollouts_with_info
from open_loop.trajectory_generator import CpgRbfNet
from open_loop.vae import VAE

LOCAL_TARJ_DIR = osp.join(LOCAL_LOG_DIR, 'Traj')

def load_AntTraj():
    traj_dir = osp.join(LOCAL_TARJ_DIR, 'Traj_Ant-v3')
    with open(osp.join(traj_dir, 'config.json'), 'r') as fp:
        config = json.load(fp)
    obs_dim, act_dim = 111, 8
    dt = 0.05
    traj_generator = CpgRbfNet(config, dt, config['num_rbf'], act_dim)

    with open(osp.join(traj_dir, 'params.pkl'), 'rb') as fp:
        traj_weight = torch.load(fp)
    traj_generator.load_state_dict(traj_weight)

    return traj_generator

def load_HalfCheetahTraj():
    traj_dir = osp.join(LOCAL_TARJ_DIR, 'Traj_HalfCheetah-v3')
    with open(osp.join(traj_dir, 'config.json'), 'r') as fp:
        config = json.load(fp)
    obs_dim, act_dim = 17, 6
    dt = 0.05
    traj_generator = CpgRbfNet(config, dt, config['num_rbf'], act_dim)

    with open(osp.join(traj_dir, 'params.pkl'), 'rb') as fp:
        traj_weight = torch.load(fp)
    traj_generator.load_state_dict(traj_weight)

    return traj_generator

def load_trajectory_generator(env_name, idx=0):
    traj_dir = osp.join(LOCAL_TARJ_DIR, f'Traj_{env_name}_{idx}')
    with open(osp.join(traj_dir, 'config.json'), 'r') as fp:
        config = json.load(fp)
    dummy_env = make_env(env_name)
    
    obs_dim, act_dim = \
        dummy_env.observation_space.shape[0], dummy_env.action_space.shape[0]
    dt = dummy_env.dt
    traj_generator = CpgRbfNet(config, dt, config['num_rbf'], act_dim)

    with open(osp.join(traj_dir, 'params.pkl'), 'rb') as fp:
        traj_weight = torch.load(fp)
    traj_generator.load_state_dict(traj_weight)

    return traj_generator

def load_vae_and_save_generated_trajs(env_name, idx, itr, n_path=11, z_max=2):
    input_sizes = {
        'Ant-v3': [21, 8],
        'HalfCheetah-v3': [21, 6],
        'MinitaurBulletEnv-v0': [21, 8],
        'MinitaurReactiveEnv-v0': [21, 8]
        }
    vae_dir = osp.join(LOCAL_TARJ_DIR, f'Vae_{env_name}_{idx}')
    with open(osp.join(vae_dir, 'config.json'), 'r') as f:
        config = json.loads(f.read())
    vae_params = torch.load(osp.join(vae_dir, f'itr_{itr}.pkl'))

    input_size = input_sizes[env_name]
    hidden_layers = config['layers']
    variable_size = config['variable_size']
    vae = VAE(input_size, variable_size, hidden_layers)
    vae.load_state_dict(vae_params)
    env = make_env(env_name, seed=config['seed'])

    z = np.linspace(-z_max, z_max, n_path)[:, None]
    latent_variables = z.repeat(config['variable_size'], 1)
    trajs = vae.generate(latent_variables)
    paths, paths_info = traj_rollouts_with_info(trajs, env)

    rollout_dir = osp.join(vae_dir, 'rollout')
    if not os.path.exists(rollout_dir):
        os.makedirs(rollout_dir)

    import pickle
    with open(osp.join(rollout_dir, f'itr_{itr}_paths'), 'wb') as f:
        pickle.dump(paths, f)

    with open(osp.join(rollout_dir, f'itr_{itr}_paths_info'), 'wb') as f:
        pickle.dump(paths_info, f)

    return z

def generate_trajectory(traj_generator, alpha=0.7):
    base_time = np.arange(0, traj_generator.period + traj_generator.timestep, traj_generator.timestep)
    base_traj = []
    for t in base_time:
        base_traj.append(traj_generator.get_action(t))
    base_traj = np.array(base_traj)
    traj_len = len(base_traj)

    gap = traj_len // 2

    init_weight = base_time / base_time[-1] # ∈ [0, 1]
    init_weight[0:gap+ 1] = alpha * init_weight[0:gap+1]
    init_weight[-gap:] = init_weight[0:gap][::-1]

    # base_traj = np.zeros(base_traj.shape)

    epsilon = np.random.randn(*base_traj.shape)

    traj = base_traj + init_weight[:, None] * epsilon

    return traj

def generate_trajectories(batch_size, *args, **kwargs):
    trajs = [generate_trajectory(*args, **kwargs) for _ in range(batch_size)]
    return trajs
