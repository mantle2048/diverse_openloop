from open_loop.scripts.plot_traj_generator_curve import *
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10
sns.set(style='whitegrid', palette='tab10', font_scale=1.5)
# %matplotlib notebook
# %reload_ext autoreload
# %autoreload 2

ENV_SET = ["Ant-v3", "HalfCheetah-v3", "MinitaurBulletEnv-v0", "MinitaurReactiveEnv-v0"]
env_name = ENV_SET[1]
exp_id = '0'

plot_train_traj_curve(env_name, 1)

plot_traj_gait_curve(env_name, exp_id)

plot_prior_distribution("HalfCheetah-v3", exp_id)

plot_x_y_position('Ant-v3',  exp_id, resample = False, n_path = 30, itr=200, with_legend=True)

plot_x_y_position('MinitaurBulletEnv-v0', exp_id, resample=False, n_path = 30, itr=40)

plot_x_y_position('MinitaurReactiveEnv-v0', 3, resample=False, n_path = 30, itr=100)

plot_x_y_velocity('Ant-v3', exp_id, resample=False, n_path = 30, itr=200)

plot_x_y_velocity('MinitaurBulletEnv-v0', exp_id, resample=False, n_path = 30, itr=40)

plot_x_y_velocity('MinitaurReactiveEnv-v0', 3, resample=False, n_path = 30, itr=100)


