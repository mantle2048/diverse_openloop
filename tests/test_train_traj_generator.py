import open_loop
from open_loop.scripts.train_trajectory_generator import get_parser, main, Traj_Trainer
# %matplotlib notebook
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

def test_run(seed=1):
    env_name = 'HalfCheetah-v3'
    
    arg_list =  [
        '--n_itr',
        '2',
        '--video_log_freq',
        '-1',
        '--seed',
        f'{seed}',
        '--amplitude',
        '1',
        '--frequency',
        '1',
        '--num_rbf',
        '100',
        '--env_name',
        f'{env_name}',
        '--exp_prefix',
        f'Traj_{env_name}',
        '--save_params',
        '--popsize',
        '16'
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)
    trainer = Traj_Trainer(config)
    trainer.run_training_loop(config['n_itr'])


if __name__ == '__main__':
    test_run()




