import os
import os.path as osp
import seaborn as sns
import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from open_loop.user_config import LOCAL_IMG_DIR, LOCAL_LOG_DIR
from open_loop.utils import load_trajectory_generator, load_vae_and_save_generated_trajs

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10
sns.set(style='whitegrid', palette='tab10', font_scale=1.5)


COLORS = (
    [
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        '#F0E442',
    ]
)

LOCAL_TARJ_DIR = osp.join(LOCAL_LOG_DIR, 'Traj')

def generate_trajectory(traj_generator, num_point = 301, alpha=0.7):

    act_dim = traj_generator.num_act
    base_time = np.linspace(0, traj_generator.period, 101)
    base_traj = traj_generator.get_action(base_time)
    traj_len = len(base_traj)

    gap = traj_len // 2

    init_weight = base_time / base_time[-1] # ∈ [0, 1]
    init_weight[0:gap+ 1] = alpha * init_weight[0:gap+1]
    init_weight[-gap:] = init_weight[0:gap][::-1]

    transpose_weight = np.random.randn(act_dim, act_dim).clip(-0.7,0.7)
    epsilon = base_traj @ transpose_weight
    traj = base_traj + init_weight[:, None] * epsilon
    traj = np.concatenate([traj, traj[1:], traj[1:]] ,axis=0)

    return traj

def generate_trajectories(batch_size, *args, **kwargs):
    trajs = [generate_trajectory(*args, **kwargs) for _ in range(batch_size)]
    return trajs

def plot_x_y_position(
    env_name,
    idx,
    itr,
    n_path=11,
    resample=False,
    with_legend=False,
):
    assert 'HalfCheetah' not in env_name, "HalfCheetah has no x_y position!"
    paths_name, paths_info_name = f'itr_{itr}_paths', f'itr_{itr}_paths_info'
    rollout_dir = osp.join(LOCAL_TARJ_DIR, f'Vae_{env_name}_{idx}', 'rollout')

    zmax = 2

    if resample or not glob.glob(osp.join(rollout_dir, paths_info_name)):
        load_vae_and_save_generated_trajs(env_name, idx, itr, n_path, z_max=zmax)
    zs = np.linspace(-zmax, zmax, n_path)

    with open(osp.join(rollout_dir, paths_info_name), 'rb') as f:
        paths_info = pickle.load(f)

    zs = range(10)
    paths_info = paths_info[12:22]

    # zs = [7]
    # paths_info = [paths_info[7]]
    # create a figure for plot
    fig, ax = plt.subplots(1, 1,)

    for z, path_info in zip(zs, paths_info):
        x_position, y_position = [], []
        for info in path_info:
            x_position.append(info['x_position'])
            y_position.append(info['y_position'] + 0.016)
        ax.plot(x_position, y_position, label=f'z={z:.2f}')

    if with_legend:
        ax.legend().set_draggable(True)
    ax.set_title(env_name)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    img_name = osp.join(LOCAL_IMG_DIR, f'xy_position-{env_name}.png')
    fig.savefig(img_name, dpi=300)

def plot_x_y_velocity(env_name, idx, itr, n_path=11, resample=False):
    assert 'HalfCheetah' not in env_name, "HalfCheetah has no x_y position!"
    paths_name, paths_info_name = f'itr_{itr}_paths', f'itr_{itr}_paths_info'
    rollout_dir = osp.join(LOCAL_TARJ_DIR, f'Vae_{env_name}_{idx}', 'rollout')
    import ipdb; ipdb.set_trace()

    if resample or not glob.glob(osp.join(rollout_dir, paths_info_name)):
        load_vae_and_save_generated_trajs(env_name, idx, itr, n_path)
    zs = np.linspace(-2, 2, n_path)

    with open(osp.join(rollout_dir, paths_info_name), 'rb') as f:
        paths_info = pickle.load(f)

    # create a figure for plot
    fig, ax = plt.subplots(2, 1)

    for z, path_info in zip(zs, paths_info):
        x_position, y_position = [], []
        for idx, info in enumerate(path_info):
            x_position.append(info['x_velocity'])
            # y_position.append(info['y_velocity'] + 0.2 * (idx / 100))
            y_position.append(info['y_velocity'] + 0.2 * (idx / 100))
        timestep = np.arange(len(x_position))
        x_position = uniform_filter1d(x_position, size=51)
        y_position = uniform_filter1d(y_position, size=51)
        ax[0].plot(timestep, x_position, label=f'z={z:.2f}')
        ax[1].plot(timestep, y_position, label=f'z={z:.2f}')

    # ax.legend()
    ax[1].set_xlabel("timestep")
    ax[0].set_ylabel("x velocity")
    ax[1].set_ylabel("y velocity")
    ax[0].set_title(env_name)
    img_name = osp.join(LOCAL_IMG_DIR, f'xy_velocity-{env_name}.png')
    fig.savefig(img_name, dpi=300)

def plot_prior_distribution(env_name, exp_id):

    # curve_color
    curve_color = [
        ('#c54f4f', '#efaaa8'), #RED Light RED
        ('#4c72b2', '#a3c9f4'), #Blue light Blut
        ('#54a86e', '#92e4a0'), #GREEN, light Green
    ]

    actuator_name = {
        "Ant-v3": ['front_left_leg_hip', 'front_left_leg_angle'],
        "HalfCheetah-v3": ['front thigh', 'front shin', 'front foot'],
        "MinitaurBulletEnv-v0" : ['front_left_leg_hip', 'front_left_leg_angle'],
        "MinitaurReactiveEnv-v0": ['front_left_leg_hip', 'front_left_leg_angle'],
        }

    # config
    batch_size = 8
    num_point = 301
    alpha = 1.0
    np.random.seed(21)

    # load trajectory generator
    traj_generator = load_trajectory_generator(env_name, exp_id)

    # create a figure for plot
    fig, ax = plt.subplots(1, 1, figsize=(9.5,5))
    ax.set_xticks([0,  100,  200, 300], ['0',  'T',  '2T',  '3T'])
    timestep = np.arange(0, num_point)

    for actuator, name in enumerate(actuator_name[env_name]):
        # get base traj
        base_time = np.linspace(0, traj_generator.period * (num_point // 100), num_point)
        base_traj = traj_generator.get_action(base_time)[:, actuator]

        # get a batch of sample from prior traj distribution
        trajs = generate_trajectories(batch_size, traj_generator, num_point, alpha)
        trajs = np.array(trajs)
        torque = trajs[:, :, actuator]

        # plot
        ax.plot(
            timestep, base_traj,
            zorder = (actuator + 1) * 10, lw = 4,
            color = curve_color[actuator][0], label = name
        )
        ax.plot(
            timestep, torque.transpose(),
            lw = 2, color = curve_color[actuator][1]
        )

    # figure config
    ax.set_xlabel("Time")
    ax.set_ylabel("Torque")
    ax.set_title(env_name)
    ax.legend(
        loc='lower center',
        ncol=4,
        handlelength=2,
        borderaxespad=0.,
        prop={'size': 14},
        mode='expand'
    )
    img_name = osp.join(LOCAL_IMG_DIR, f'prior-dist.png')
    fig.savefig(img_name, dpi=300)

def plot_traj_gait_curve(env_name, exp_id):

    torque_group = {
        "Ant-v3": ((0, 2), (2, 4), (4, 6),(6,8)),
        "HalfCheetah-v3": ((0, 3), (3, 6)),
        "MinitaurBulletEnv-v0" : ((0, 2), (2, 4), (4, 6), (6,8)),
        "MinitaurReactiveEnv-v0": ((0, 2), (2, 4), (4, 6), (6,8)),
    }
    legend_group = {
        "Ant-v3": (
            ['front_left_leg_hip', 'front_left_leg_angle'],
            ['front_right_leg_hip', 'front_right_leg_angle'],
            ['back_left_leg_hip', 'back_left_leg_angle'],
            ['back_right_leg_hip', 'back_right_leg_angle'],
        ),
        "HalfCheetah-v3": (
            ['front thigh', 'front shin', 'front foot'],
            ['back thigh', 'back shin', 'back foot'],
        ),
        "MinitaurBulletEnv-v0" : (
            ['front_left_leg_hip', 'front_left_leg_angle'],
            ['front_right_leg_hip', 'front_right_leg_angle'],
            ['back_left_leg_hip', 'back_left_leg_angle'],
            ['back_right_leg_hip', 'back_right_leg_angle'],
        ),
        "MinitaurReactiveEnv-v0": (
            ['front_left_leg_hip', 'front_left_leg_angle'],
            ['front_right_leg_hip', 'front_right_leg_angle'],
            ['back_left_leg_hip', 'back_left_leg_angle'],
            ['back_right_leg_hip', 'back_right_leg_angle'],
        ),
        }

    curve_color = [
        '#c54f4f',#RED
        '#4c72b2',#Blue
        '#54a86e',#GREEN
    ]

    traj_generator = load_trajectory_generator(env_name, exp_id)
    num_point = 301
    timestep = np.arange(0, num_point)
    t = np.linspace(0, traj_generator.period * (num_point // 100), num_point)
    act = traj_generator.get_action(t)

    # create the figure
    i = 0

    for (a, b), legend in zip(torque_group[env_name], legend_group[env_name]):
        fig, ax  = plt.subplots(1, figsize=(12,6))
        for _ in range(a, b):
            ax.plot(timestep, act[:, _], linewidth=3, color = curve_color[_ % 3])
        ax.set_xticks([0,  100,  200, 300], ['0',  'T',  '2T',  '3T'])
        ax.set_xlabel("Time")
        ax.set_ylabel("Torque")
        ax.set_title(env_name)
        ax.legend(
            legend,
            loc='lower center',
            ncol=4,
            handlelength=2,
            borderaxespad=0.,
            prop={'size': 14},
            mode='expand'
        )
        img_name = osp.join(LOCAL_IMG_DIR, f'gail-{i}.png')
        fig.savefig(img_name, dpi=300)
        i += 1

def plot_train_traj_curve(env_name, exp_id):

    smooth_size = {
        "Ant-v3": 5,
        "HalfCheetah-v3": 3,
        "MinitaurBulletEnv-v0" : 1,
        "MinitaurReactiveEnv-v0": 1
    }

    ylim_size = {
        "Ant-v3": (180, 520),
        "HalfCheetah-v3": (0, 400),
        "MinitaurBulletEnv-v0" : (0, 4.2),
        "MinitaurReactiveEnv-v0": (0, 2.8)
    }

    traj_dir = osp.join(LOCAL_TARJ_DIR, f'Traj_{env_name}_{exp_id}')
    progress = pd.read_csv(osp.join(traj_dir, 'progress.csv'), sep=',')

    # create the figure
    fig, ax = plt.subplots(1, figsize=(8,6))

    # plot best return
    itr = progress['Itr'].to_numpy()
    best_return = progress['BestReturn'].to_numpy()
    ax.plot(
        itr,
        best_return,
        linewidth = 3.0,
        color = COLORS[0],
    )

    # plot fill_between eval reward

    smooth = smooth_size[env_name]
    eval_return = uniform_filter1d(progress['AverageTrainReward'].to_numpy(), size=smooth)
    max_eval_return = uniform_filter1d(progress['MaxTrainReward'].to_numpy(), size=smooth)
    min_eval_return = uniform_filter1d(progress['MinTrainReward'].to_numpy(), size=smooth)

    sns.lineplot(
        x = np.concatenate([itr, itr, itr]),
        y = np.concatenate([eval_return, max_eval_return, min_eval_return]),
        ci = 'sd',
        linewidth = 3.0,
        color = COLORS[2],
        ax = ax,
    )

    # figure config
    ax.margins(x=0.0, y=0.03)
    ax.set_xlim(0, 500)
    ax.set_title(env_name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Return')

    # bottom, top = ylim_size[env_name]
    # ax.set_ylim(bottom=bottom, top=top)
    ax.legend(['BestReturn', 'AverageReturn'], loc='best')
    img_path = osp.join(LOCAL_IMG_DIR, f'{env_name}-traj_train_curve.png')
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
