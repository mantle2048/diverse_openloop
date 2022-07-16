# -*- coding: utf-8 -*-
# +
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from open_loop.transfer import *
from open_loop.vae import *
from open_loop.utils import *
from reRLs.infrastructure.utils.utils import Path, write_gif

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

env_set = ["HalfCheetah-v3", "Ant-v3", "MinitaurBulletEnv-v0"] 
alpha_set = [0.7, 0.7, 1.2]


def plot_traj_curve(env_name, idx=0):
    traj = load_trajectory_generator(env_name, idx)
    act_dim = traj.num_act
    length = 101
    x = np.linspace(0, traj.period, length)
    y = []
    for t in x:
        y.append(traj.get_action(t))
    y = np.array(y)

    robot_x = np.arange(0, traj.period + traj.timestep, traj.timestep)
    robot_y = []
    for t in robot_x:
        robot_y.append(traj.get_action(t))
    robot_y = np.array(robot_y)

    fig, axs = plt.subplots(act_dim // 2, 2, figsize=(12,8))
    axs = axs.flatten()
    for idx, ax in enumerate(axs):
        ax.plot(x, y[:, idx], linewidth=3, zorder=0)
        ax.scatter(robot_x, robot_y[:, idx], color='r', marker = 'o', zorder=1)
        ax.set_title(f"Act: {idx}")
plot_traj_curve(env_set[2], 0)


def plot_traj_dist_scatter(env_name, idx=0, alpha=0.7):
    traj = load_trajectory_generator(env_name, idx)
    act_dim = traj.num_act
    fig, axs = plt.subplots(act_dim // 2, 2, figsize=(12,8))
    axs = axs.flatten()

    x = np.linspace(0, traj.period, 101)
    init_traj = []
    for t in x:
        init_traj.append(traj.get_action(t))
    init_traj = np.array(init_traj)
    for idx, ax in enumerate(axs):
        ax.plot(x, init_traj[:, idx], linewidth=3, zorder=1)
        ax.set_title(f"Act: {idx}")

    for _ in range(10):
        generated_traj = generate_trajectory(traj, alpha=alpha)
        x = np.arange(0, traj.period + traj.timestep, traj.timestep)
        for idx, ax in enumerate(axs):
            ax.scatter(x, generated_traj[:, idx], color='r', marker = 'o', zorder=0)
id = 2
plot_traj_dist_scatter(env_set[id], 0, alpha_set[id])


def plot_traj_dist_curve(env_name, idx=0, actuator=0, alpha = 0.7):
    traj = load_trajectory_generator(env_name, idx)
    fig, ax = plt.subplots(1, 1, figsize=(8,4.3))
    x = np.linspace(0, traj.period, 101)
    init_traj = []
    for t in x:
        init_traj.append(traj.get_action(t))
    init_traj = np.array(init_traj)

    ax.plot(x, init_traj[:, actuator], color='r', linewidth=3, zorder=1)
    ax.legend(['Initial Trajectory'])
    ax.set_title(f"actuator: {0}")

    for _ in range(70):
        generated_traj = generate_trajectory(traj, alpha=alpha) 
        x = np.arange(0, traj.period + traj.timestep, traj.timestep)
        ax.plot(x, generated_traj[:, actuator], marker = 'o', zorder=0)

    ax.set_xlabel("Time")
    ax.set_ylabel("Torque")
plot_traj_dist_curve(env_set[2], actuator=1)

# +
fig, ax = plt.subplots(1, 1, figsize=(8,4.3))

x = np.linspace(0, ant_traj.period, 101)
init_traj = []
for t in x:
    init_traj.append(ant_traj.get_action(t))
init_traj = np.array(init_traj)

ax.plot(x, init_traj[:, 0], color='r', linewidth=3, zorder=1)
ax.legend(['Initial Trajectory'])
ax.set_title(f"actuator: {0}")

for _ in range(70):
    traj = generate_trajectory(ant_traj) 
    x = np.arange(0, ant_traj.period + ant_traj.timestep, ant_traj.timestep)
    ax.plot(x, traj[:, 4], marker = 'o', zorder=0)

ax.set_xlabel("Time")
ax.set_ylabel("Torque")
# -

from pyvirtualdisplay import Display
virtual_disp = Display(visible=False, size=(1400, 900))


def generate(traj_generator, vae):
    fig, axs = plt.subplots(4, 2, figsize=(12,8))
    axs = axs.flatten()
    x = np.linspace(0, traj_generator.period, 101)
    y = []
    for i in x:
        y.append(traj_generator.get_action(i))
    y = np.array(y)
    for idx, ax in enumerate(axs):
        ax.plot(x, y[:, idx], linewidth=3, zorder=0)
        ax.set_title(f"Act: {idx}")
        
    env = gym.make("Ant-v3")
    
    zs = torch.linspace(-2, 2, 11)
    zs = torch.cat([zs ,zs]).reshape(11,2)
    print(zs)
    vae.eval()
    for i in zs:
        
        # traj = generate_trajectory(traj_generator).flatten()
        # with torch.no_grad():
            # reconstruct_x = vae.generate(traj).squeeze()
        z = torch.randn(2)
        # z = i[None]
        with torch.no_grad():
            reconstruct_x = vae.decoder(z[None]).squeeze()
            
        point_num = int(traj_generator.period / traj_generator.timestep) + 1
        act_num = traj_generator.num_act
        traj = reconstruct_x.view(point_num, act_num)
        traj = traj.cpu().numpy()
        
        rendered_path = rollout(env, traj, length=500, render=True)
        rendered_path_rew = rendered_path['rew'].sum()
        write_gif(f'vae_z_{i.item(): 0.2f}_rew_{rendered_path_rew}', rendered_path['image_obs'], fps=30)
       
        paths = rollouts(5, env, traj)
        rollout_rew = np.mean([path['rew'].sum() for path in paths])
        
        print('rollout rew:', rollout_rew)
        t = np.linspace(0, traj_generator.period, point_num)

        for idx, ax in enumerate(axs):
            ax.scatter(t, traj[:, idx], color='r', zorder=0)
            ax.set_title(f"Act: {idx}")


saved_vae = VAE(21 * 8, 2, [256, 512, 256], 0.001)
saved_vae.load_state_dict(torch.load('../vae_10.pkl'))
generate(ant_traj, saved_vae)

vae
