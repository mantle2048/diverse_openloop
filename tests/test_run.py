import open_loop
from open_loop.scripts.run import get_parser, main, Traj_Trainer
# %matplotlib notebook
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

def test_run(seed=1):
    env_name = 'HalfCheetah-v3'
    
    arg_list =  [
        '--n_itr',
        '101',
        '--video_log_freq',
        '20',
        '--seed',
        f'{seed}',
        '--amplitude',
        '1.0',
        '--frequency',
        '5',
        '--num_rbf',
        '100',
        '--env_name',
        f'{env_name}',
        '--exp_prefix',
        f'Traj_{env_name}',
        '--save_params',
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)
    trainer = Traj_Trainer(config)
    trainer.run_trainning_loop(config['n_itr'])


if __name__ == '__main__':
    test_run()


