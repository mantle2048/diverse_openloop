import os
import os.path as osp
import torch
import json
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import open_loop.user_config as conf

from typing import Dict
from collections import defaultdict

from torch import nn
from open_loop.trajectory_generator import CpgRbfNet
from open_loop.user_config import LOCAL_LOG_DIR
from open_loop.envs.make_env import make_env
from open_loop.rollout import rollout, traj_rollouts
from open_loop.vae import VAE
from reRLs.infrastructure.loggers import setup_logger

# %matplotlib notebook
# %reload_ext autoreload
# %autoreload 2
# -

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

def generate_trajectories(batch_size, *args, **kwargs):
    trajs = [generate_trajectory(*args, **kwargs) for _ in range(batch_size)]
    return trajs

class Latent_Trainer():

    def __init__(self, config: Dict):

        self.config = config

        assert config['env_name'] in ("HalfCheetah-v3", "Ant-v3")
        self.env = make_env(config['env_name'], config['seed'])

        # x y z:
        # self.env.sim.model.cam_poscom0[0] = np.array([0., -3., 1.7])
        # self.env.sim.model.cam_fovy[0] = 20
        # self.env.sim.model.cam_targetbodyid[0] = 0
        # self.env.sim.model.cam_bodyid[0] = 5
        self.env.sim.model.cam_quat[0][2] = 1.0

        self.logger = setup_logger(
            exp_prefix=config['exp_prefix'],
            seed=config['seed'],
            exp_id=config['exp_id'],
            snapshot_mode=config['snapshot_mode'],
            base_log_dir=config['base_log_dir']
        )

        self.traj_generator =  \
            load_trajectory_generator(config['env_name'], config['exp_id'])

        num_point = int(self.traj_generator.period / self.traj_generator.timestep) + 1
        act_dim = self.traj_generator.num_act

        self.vae = VAE(
            input_size = [num_point, act_dim],
            variable_size = config['variable_size'],
            hidden_layers= config['layers'],
            learning_rate = config['lr'],
        )

        # Set random seed (must be set after es_sovler)
        seed = self.config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.batch_size = self.config['batch_size']

        self.fps = 30
        self.config['fps'] = self.fps

    def run_training_loop(self, n_itr):

        self.start_time = time.time()
        train_logs = defaultdict(list)

        for itr in range(n_itr):
            ## decide if tabular should be logged
            self._refresh_logger_flags(itr)

            trajs = generate_trajectories(self.batch_size, self.traj_generator)
            trajs = np.array(trajs)

            rew_list = []
            train_log = self.vae.update(trajs, rew_list)
            for key, val in train_log.items():
                train_logs[key].append(val)

            ## log/save
            if self.logtabular:
                ## perform tabular and video
                self.perform_logging(itr, train_log)

                if self.config['save_params'] and self.logparam:
                    self.logger.save_itr_params(itr, self.vae.get_state())

        fig, axs = plt.subplots(len(train_logs.keys()), 1, figsize=(8,15))
        axs = axs.flatten()
        for ax, key in zip(axs, train_logs.keys()):
            loss = train_logs[key]
            x = np.arange(0, len(loss))
            ax.plot(x, loss, linewidth=3, zorder=0)
            ax.set_title(f"Loss: {key}")

        figure_dir = osp.join(self.logger._snapshot_dir, 'train_loss.png')
        fig.savefig(figure_dir, dpi=200)

        self.env.close()
        self.logger.close()

    def perform_logging(self, itr, train_log):

        if itr == 0:
            self.logger.log_variant('config.json', self.config)

        if self.logvideo:
            n_path = 1
            z = np.linspace(-2, 2, n_path)[:, None]
            latent_variables = z.repeat(self.config['variable_size'], 1)
            trajs = self.vae.generate(latent_variables)
            video_paths = traj_rollouts(trajs, self.env, render=True)
            self.logger.log_paths_as_videos(
                video_paths, itr,
                max_videos_to_save = len(video_paths),
                fps=self.fps, video_title='rollout'
            )

        self.logger.record_tabular("Itr", itr)
        self.logger.record_tabular("Time", (time.time() - self.start_time) / 60)
        self.logger.record_dict(train_log)

        self.logger.dump_tabular(with_prefix=True, with_timestamp=False)

    def _refresh_logger_flags(self, itr):

        if self.config['tabular_log_freq'] != -1 \
                and itr % self.config['tabular_log_freq'] == 0:
            self.logtabular = True
        else:
            self.logtabular = False

        if itr == self.config['n_itr'] - 1:
            self.logvideo = True
        else:
            self.logvideo = False

        if self.config['param_log_freq'] != -1 \
                and itr % self.config['param_log_freq'] == 0 \
                    and self.config['save_params']:
            self.logparam = True
        else:
            self.logparam = False


def get_parser():

    import argparse
    parser = argparse.ArgumentParser()

    # exp args
    parser.add_argument('--env_name', type=str, default='Ant-v3')

    # logger args
    parser.add_argument('--exp_prefix', type=str, default='Vae_Ant-v3')
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="gap_and_last")
    parser.add_argument('--base_log_dir', type=str, default=f"{conf.LOCAL_LOG_DIR}")

    # train args
    parser.add_argument('--n_itr', '-n', type=int, default=101)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--tabular_log_freq', type=int, default=1)
    parser.add_argument('--param_log_freq', type=int, default=10)
    parser.add_argument('--save_params', action='store_true')

    # vae args
    parser.add_argument("--variable_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--layers", nargs='+', type=int, default=[256, 512, 256])

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    trainer = Traj_Trainer(config)
    trainer.run_training_loop(config['n_itr'])

if __name__ == '__main__':
    main()
