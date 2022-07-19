from open_loop.scripts.plot_traj_generator_curve import *
# %matplotlib notebook
# %reload_ext autoreload
# %autoreload 2

ENV_SET = ["Ant-v3", "HalfCheetah-v3", "MinitaurBulletEnv-v0", "MinitaurReactiveEnv-v0"]
env_name = ENV_SET[0]
exp_id = '0'

plot_train_traj_curve(env_name, exp_id)

plot_traj_gait_curve(env_name, exp_id)

plot_prior_distribution(env_name, exp_id)

base_traj = np.random.randn(301, 8)
get_prior_dist(base_traj)




