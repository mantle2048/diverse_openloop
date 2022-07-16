# +
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from open_loop.vae import *
from reRLs.infrastructure.utils.utils import Path, write_gif
from open_loop.scripts.train_latent_space import get_parser, main, Latent_Trainer

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10
sns.set(style='whitegrid', palette='tab10', font_scale=1.5)


# %matplotlib notebook
# %reload_ext autoreload
# %autoreload 2
# -

def test_trainer(seed=1):
    env_set = [
        "HalfCheetah-v3",
        "Ant-v3",
        "MinitaurBulletEnv-v0",
        "AntBulletEnv-v0",
        'cheetah_run',
        'MinitaurTrottingEnv-v0',
        'MinitaurReactiveEnv-v0',
    ]
    env_name = env_set[2]
    arg_list =  [
        '--alpha',
        '0.7',
        '--n_itr',
        '11',
        '--exp_id',
        '0',
        '--video_log_freq',
        '1',
        '--seed',
        '1',
        '--lr',
        '0.001',
        '--n_path',
        '11',
        '--env_name',
        f'{env_name}',
        '--exp_prefix',
        f'Vae_{env_name}',
        '--save_params'
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)
    trainer = Latent_Trainer(config)
    trainer.run_training_loop(config['n_itr'])


if __name__ == '__main__':
    test_trainer()


