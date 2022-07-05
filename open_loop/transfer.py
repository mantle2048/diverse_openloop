import os
import os.path as osp
import torch
import json
import gym
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from open_loop.user_config import LOCAL_LOG_DIR
from open_loop.trajectory_generator import CpgRbfNet
from open_loop.vae import VAE
from reRLs.infrastructure.utils.utils import write_gif, Path

LOCAL_TARJ_DIR = osp.join(LOCAL_LOG_DIR, 'Traj')

def load_AntTraj():
    traj_dir = osp.join(LOCAL_TARJ_DIR, 'Traj_Ant-v3_0')
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
    traj_dir = osp.join(LOCAL_TARJ_DIR, 'Traj_HalfCheetah-v3_0')
    with open(osp.join(traj_dir, 'config.json'), 'r') as fp:
        config = json.load(fp)
    obs_dim, act_dim = 17, 6
    dt = 0.05
    traj_generator = CpgRbfNet(config, 0.05, config['num_rbf'], act_dim)

    with open(osp.join(traj_dir, 'params.pkl'), 'rb') as fp:
        traj_weight = torch.load(fp)
    traj_generator.load_state_dict(traj_weight)

    return traj_generator

def rollout(env, traj, length=200, seed=0,render=False):
    obs = env.reset()
    obss, acts, rews, next_obss, terminals, image_obss = [], [], [], [], [], []
    for i in range(length):
        act = traj[i % len(traj)].clip(-1, 1)
        # act = traj[i % len(traj)]
        if render:
            image_obss.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
        obss.append(obs)
        next_obs, rew, done, _ = env.step(act)
        rews.append(rew)

        next_obss.append(next_obs)
        terminals.append(done)

        if done:
            break

    return Path(obss, image_obss, acts, rews, next_obss, terminals)

def rollouts(n_path, *args, **kwargs):
    paths = [rollout(*args, **kwargs) for _ in range(n_path)]
    return paths

def generate_trajectory(traj_generator):
    base_time = np.arange(0, traj_generator.period + traj_generator.timestep, traj_generator.timestep)
    base_traj = []
    for t in base_time:
        base_traj.append(traj_generator.get_action(t))
    base_traj = np.array(base_traj)
    traj_len = len(base_traj)

    gap = traj_len // 2
    alpha = 0.7

    init_weight = base_time / base_time[-1] # ∈ [0, 1]
    init_weight[0:gap+ 1] = alpha * init_weight[0:gap+1]
    init_weight[-gap:] = init_weight[0:gap][::-1]

    # base_traj = np.zeros(base_traj.shape)

    epsilon = np.random.randn(*base_traj.shape)

    traj = base_traj + init_weight[:, None] * epsilon

    return traj

def generate_trajectories(n_traj, traj_generator):
    trajs = [generate_trajectory(traj_generator) for _ in range(n_traj)]
    return trajs


def run_training_loop(env, traj_generator):
    from collections import defaultdict
    train_logs = defaultdict(list)
    num_point = int(traj_generator.period / traj_generator.timestep) + 1
    act_dim = traj_generator.num_act
    vae = VAE(
        input_size = num_point * act_dim,
        variable_size = 2,
        hidden_layers=[256, 512, 256],
        learning_rate = 0.001
    )
    batch_size = 32
    for i in range(1, 101):
        rew_list = []
        trajs = generate_trajectories(batch_size, traj_generator)
        trajs = np.array(trajs)
        for traj in trajs:
            paths = rollouts(10, env, traj)
            rew = np.mean([path['rew'].sum() for path in paths])
            rew_list.append(rew)
        print(np.mean(rew_list))
        train_log = vae.update(trajs, rew_list)
        for key, val in train_log.items():
            train_logs[key].append(val)
        print(train_log)

        if i % 10 == 0:
            fig, axs = plt.subplots(3, 1, figsize=(12,8))
            axs = axs.flatten()
            for ax, key in zip(axs, train_logs.keys()):
                loss = train_logs[key]
                x = np.arange(0, len(loss))
                ax.plot(x, loss, linewidth=3, zorder=0)
                ax.set_title(f"Loss: {key}")
            fig.savefig('train_loss.png', dpi=200)

            torch.save(vae.state_dict(), f'vae_{i}.pkl')
    return train_logs, vae


if __name__ == '__main__':
    env = gym.make("Ant-v3")
    ant_traj = load_AntTraj()
    run_training_loop(env, ant_traj)

