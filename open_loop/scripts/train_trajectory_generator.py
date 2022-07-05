# +
import numpy as np
import torch
import time
import gym
import random

import open_loop
import open_loop.user_config as conf

from typing import Dict
from pyvirtualdisplay import Display
from matplotlib import pyplot as plt

from open_loop.cma_es import CMAES 
from reRLs.infrastructure.loggers import setup_logger
from open_loop.trajectory_generator import CpgRbfNet
from open_loop.rollout import WorkerSet, serial_sample, parallel_sample, rollout, rollouts
from open_loop.envs.make_env import make_env
from open_loop.envs.wrappers.trajectory_generator_wrapper_env import TrajectoryGeneratorWrapperEnv

# %matplotlib notebook
# %reload_ext autoreload
# %autoreload 2
# -

class Traj_Trainer():

    def __init__(self, config: Dict):
        self.config = config

        self.env = make_env(config['env_name'], config['seed'])
        self.config['num_act'] = self.env.action_space.shape[0]
        self.config['timestep'] = self.env.dt

        sin_config = {
            'amplitude': config['amplitude'],
            'theta': config['theta'],
            'frequency': config['frequency'],
        }
        self.trajectory_generator = CpgRbfNet(
            sin_config, config['timestep'], config['num_rbf'], config['num_act']
        )

        self.env = TrajectoryGeneratorWrapperEnv(
            self.env, self.trajectory_generator
        )

        self.logger = setup_logger(
            exp_prefix=config['exp_prefix'],
            seed=config['seed'],
            exp_id=config['exp_id'],
            snapshot_mode=config['snapshot_mode'],
            base_log_dir=config['base_log_dir']
        )

        self.virtual_disp = Display(visible=False, size=(1400,900))
        self.virtual_disp.start()

        self.es_solver = CMAES(
            num_params=self.trajectory_generator.num_params,
            popsize=self.config['popsize'],
            sigma_init=0.01,
        )

        # Set random seed (must be set after es_sovler)
        seed = self.config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.worker_set = WorkerSet(self.config['popsize'], self.env, self.config)

        # simulation timestep, will be used for video saving
        self.fps=30
        self.config['fps'] = self.fps

    def run_training_loop(self, n_itr):

        self.start_time = time.time()

        for itr in range(n_itr):

            ## decide if videos should be rendered/logged at this iteration
            if self.config['video_log_freq'] != -1 \
                    and itr % self.config['video_log_freq'] == 0:
                self.logvideo = True
            else:
                self.logvideo = False

            ## decide if tabular should be logged
            if self.config['tabular_log_freq'] != -1 \
                    and itr % self.config['tabular_log_freq'] == 0:
                self.logtabular = True
            else:
                self.logtabular = False

            solutions = self.es_solver.ask()
            self.worker_set.sync_weights(solutions)
            train_log_dict = parallel_sample(self.worker_set)
            self.es_solver.tell(train_log_dict['ep_rews'])

            # first element is the best solution, second element is the best fitness
            best_param, best_fitness, _, _ = self.es_solver.result()

            if self.logtabular:
                self.perform_logging(itr, best_param, best_fitness, train_log_dict)

                if self.config['save_params']:
                    self.logger.save_itr_params(itr, self.trajectory_generator.get_state())

        self.env.close()
        self.worker_set.close()
        self.logger.close()

    def perform_logging(self, itr, best_param, best_fitness, train_log_dict):

        if itr == 0:
            self.logger.log_variant('config.json', self.config)

        self.worker_set.local_worker.set_weight(best_param)

        eval_log_dict = serial_sample(self.worker_set) # serial sample from local worker

        if self.logvideo:
            video_paths = rollouts(2, self.worker_set.local_worker.env, render=True)
            self.logger.log_paths_as_videos(
                video_paths, itr, fps=self.fps, video_title='rollout'
            )

            fig = plt.figure(figsize=(6,4))
            ax_1 = self.trajectory_generator.cpg.plot_curve(fig.add_subplot(121))
            ax_2 = self.trajectory_generator.plot_curve(fig.add_subplot(122))
            self.logger.log_figure(fig, 'trajectory_curve', itr)

        self.logger.record_tabular("Itr", itr)
        self.logger.record_tabular("TotalEnvInteracts", itr)

        self.logger.record_tabular_misc_stat("TrainReward", train_log_dict['ep_rews'])
        self.logger.record_tabular_misc_stat("EvalReward", eval_log_dict['ep_rews'])
        self.logger.record_tabular("BestReturn", best_fitness)
        self.logger.record_tabular("TrainEpLen", np.mean(train_log_dict['ep_lens']))
        self.logger.record_tabular("EvalEpLen", np.mean(eval_log_dict['ep_lens']))
        self.logger.record_tabular("Time", (time.time() - self.start_time) / 60)

        self.logger.dump_tabular(with_prefix=True, with_timestamp=False)

def get_parser():

    import argparse
    parser = argparse.ArgumentParser()

    # exp args
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v3')

    # logger args
    parser.add_argument('--exp_prefix', type=str, default='Traj_HalfCheetah-v3')
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="last")
    parser.add_argument('--base_log_dir', type=str, default=f"{conf.LOCAL_LOG_DIR}")

    # train args
    parser.add_argument('--n_itr', '-n', type=int, default=10)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--tabular_log_freq', type=int, default=1)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--n_path',  type=int, default=5, help="the num of trajs to eval solution")

    # cpg_rbf args
    parser.add_argument('--amplitude', '-A', type=float, default=0.2)
    parser.add_argument('--theta', type=float, default=-0.5*np.pi)
    parser.add_argument('--frequency', type=float, default=1.0)
    parser.add_argument('--num_rbf', type=int, default=20)

    # es args
    parser.add_argument('--popsize', type=int, default=10)

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    config = vars(args)

    trainer = Traj_Trainer(config)
    trainer.run_training_loop(config['n_itr'])

if __name__ == '__main__':
    main()

