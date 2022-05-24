import open_loop
from open_loop.scripts.run import get_parser, main, Traj_Trainer
# %matplotlib notebook
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

def test_run(seed=1):
    
    arg_list =  [
        '--n_itr',
        '11',
        '--video_log_freq',
        '10',
        '--seed',
        f'{seed}',
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)
    trainer = Traj_Trainer(config)
    trainer.run_trainning_loop(config['n_itr'])


if __name__ == '__main__':
    test_run()


