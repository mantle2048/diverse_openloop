import os
import os.path as osp
import torch
import numpy as np
import gym
import inspect
import open_loop
import pickle

from collections import OrderedDict
from torch import nn
from copy import deepcopy
from reRLs.infrastructure.utils import pytorch_util as ptu
from reRLs.infrastructure.utils.utils import Path
from matplotlib import pyplot as plt

PROJDIR = osp.dirname(osp.dirname(inspect.getfile(open_loop)))
RENDER = False
SEED = 0

class Agent(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        self.pi = ptu.build_mlp(
            input_size=obs_dim,
            output_size=act_dim,
            layers=[400, 300],
            activation='relu',
        )

        self.q1 = ptu.build_mlp(
            input_size=obs_dim+act_dim,
            output_size=1,
            layers=[400, 300],
            activation='relu',
        )

        self.q2 = ptu.build_mlp(
            input_size=obs_dim+act_dim,
            output_size=1,
            layers=[400, 300],
            activation='relu',
        )

    def get_action(self, obs: np.ndarray):

        if len(obs.shape) == 1:
            obs = obs[None]

        obs = ptu.from_numpy(obs.astype(np.float32))

        act = self.pi(obs)
        act = torch.clip(act, -1, 1)

        return ptu.to_numpy(act)



def get_model_state_dict(proj_dir, env_name, model_id: int=1):

    assert model_id >= 1 and model_id <= 10, 'model id error'
    model_name = f'model{model_id}.pth'
    model_dir = osp.join(
        proj_dir,
        'data',
        'pretrained_model',
        env_name,
        model_name
    )
    model_state_dict = torch.load(model_dir, map_location='cpu')
    model_state_dict = drop_dummpy_param(model_state_dict)

    print("==============" * 5)
    print(f"Load state_dict from {model_dir}")


    return model_state_dict

def drop_dummpy_param(model_state_dict: OrderedDict):
    state_dict = deepcopy(model_state_dict)
    for name, params in state_dict.items():
        if 'expert' in name:
            model_state_dict.pop(name)
        else:
            new_name = '.'.join([name.split('.')[0], name.split('.')[-2] ,name.split('.')[-1]])
            model_state_dict[new_name] = model_state_dict.pop(name)
    return model_state_dict

def collect_path(policy, env, render=False):

    obss, acts, rews, next_obss, terminals, image_obss = [], [], [], [], [], []
    obs = env.reset()
    steps = 0
    while True:
        if render:
            image_obss.append(self._env.render(mode='rgb_array'))
        obss.append(obs)
        act = policy.get_action(obs)
        if len(act.shape) > 1:
            act = act[0]
        acts.append(act)

        next_obs, rew, done, _ = env.step(act)

        rews.append(rew)
        next_obss.append(next_obs)

        steps += 1

        rollout_done = done
        terminals.append(rollout_done)
        obs = next_obs

        if rollout_done:
            break


    return Path(obss, image_obss, acts, rews, next_obss, terminals)

def collect_and_save(env_names, model_ids):

    for env_name in env_names:
        for model_id in model_ids:

            env = gym.make(env_name)
            env.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            model_state_dict = get_model_state_dict(PROJDIR, env_name, model_id)
            agent = Agent(obs_dim=obs_dim, act_dim=act_dim)
            agent.load_state_dict(model_state_dict)

            path = collect_path(policy=agent, env=env, render=RENDER)

            rollout_name = f'{env_name}_{model_id}.pkl'
            rollout_dir = osp.join(PROJDIR, 'trajs', rollout_name)

            with open(rollout_dir, 'wb+') as fp:
                pickle.dump(path, fp)

            print(f"Collect a trajectory of {env_name}, epsiode reward is {np.sum(path['rew'])}")
            print(f"Save the collected trajectory to {rollout_dir}")
            print("==============" * 5)


def main():
    env_names = [
        # 'HalfCheetah-v3',
        # 'Ant-v3',
        'Hopper-v3',
        'Walker2d-v3'
    ]
    model_ids = range(1, 11)
    # collect_and_save(env_names, model_ids)


    save_figs(env_names, model_ids)

def plot_act_curve(path_name, path):
    acts = path['act']
    length = 100
    x = np.arange(0, length)
    n_cols = 2
    n_rows = int(len(acts[0])) // n_cols
    if "Hopper" in path_name: n_rows += 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14,8))
    fig_title = f'{path_name}: {np.sum(path["rew"])}'
    fig.suptitle(fig_title)
    axs = axs.flatten()

    for i in range(len(acts[0])):
        ax = axs[i]
        ax.plot(x, acts[:length, i])
    fig_name = f'{path_name}.png'
    img_dir = osp.join(PROJDIR, 'trajs', 'act_curve')
    fig_dir = osp.join(img_dir, fig_name)
    fig.savefig(fig_dir, dpi=200)
    print("==============" * 5)
    print(f"Save the act_plot to {fig_dir}")
    print("==============" * 5)

def save_figs(env_names, traj_ids):
    traj_dir = osp.join(PROJDIR, 'trajs')
    for env_name in env_names:
        for traj_id in traj_ids:
            traj_name = f'{env_name}_{traj_id}.pkl'
            traj = osp.join(traj_dir, traj_name)
            with open(traj, 'rb') as fp:
                path = pickle.load(fp)
            path_name = osp.splitext(traj_name)[0]
            plot_act_curve(path_name, path)














