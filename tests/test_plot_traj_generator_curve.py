from open_loop.scripts.plot_traj_generator_curve import *
# %matplotlib notebook
# %reload_ext autoreload
# %autoreload 2

ENV_SET = ["Ant-v3", "HalfCheetah-v3", "MinitaurBulletEnv-v0", "MinitaurReactiveEnv-v0"]
env_name = ENV_SET[1]
exp_id = '0'

plot_train_traj_curve(env_name, exp_id)

plot_traj_gait_curve(env_name, exp_id)

plot_prior_distribution("HalfCheetah-v3", exp_id)

plot_x_y_position('Ant-v3',  exp_id, resample=False, n_path = 30, itr=200)

plot_x_y_position('MinitaurBulletEnv-v0', exp_id, resample=False, n_path = 32, itr=100)

plot_x_y_position('MinitaurReactiveEnv-v0', 1, resample=False, n_path = 30, itr=100)

plot_x_y_velocity('Ant-v3', exp_id, resample=False, n_path = 32, itr=200)
