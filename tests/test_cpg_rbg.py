# +
import numpy as np

import matplotlib.pyplot as plt
import pickle
import unittest
import pytest
from tqdm import tqdm
import numpy as np
import torch

from motion_imitation.envs.env_builder import build_regular_env
from motion_imitation.robots import a1
from motion_imitation.robots import robot_config
from motion_imitation.robots import laikago

from reRLs.infrastructure.utils import pytorch_util as ptu
from reRLs.infrastructure.utils.utils import write_gif

from es import SimpleGA, CMAES, PEPG, OpenES

import open_loop
from open_loop.meta_openloop import CentralPatternGeneratorNetwork, RadialBasisFunctionNetwork, CpgRbfNet

# %matplotlib notebook
# %reload_ext autoreload
# %autoreload 2
# -

sin_config = {'amplitude':0.2, 'theta': -0.5 * np.pi, 'frequency': 1.0}
cpg = CentralPatternGeneratorNetwork(sin_config, timestep=0.01)

cpg.reset()
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
cpg.plot_curve(ax)

rbf = RadialBasisFunctionNetwork(num_rbf=9, cpg_net=cpg)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
rbf.plot_curve(ax, cpg)

sin_config = {'amplitude':0.2, 'theta': -0.5 * np.pi, 'frequency': 1.0}
timestep = 0.01
num_rbf = 20
num_act = 3
cpg_rbfn = CpgRbfNet(sin_config, timestep, num_rbf, num_act)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
cpg_rbfn.plot_curve(ax)

env = build_regular_env(
    a1.A1,
    motor_control_mode=robot_config.MotorControlMode.POSITION,
    enable_rendering=False,
    action_limit=0.75,
    wrap_trajectory_generator=True,
    on_rack=False
)


def rollout(env, policy):
    obs = env.reset()
    cpg_rbfn.reset()
    path_return = 0.
    
    for step in range(100):
        act = cpg_rbfn.get_action(obs)
        act[0] = 0
        total_act = np.concatenate([act, act, -act, -act])
        next_obs, rew, done, info = env.step(total_act)
        path_return += rew
        if done: break
    return path_return


NPARAMS = len(cpg_rbfn.get_flat_weight())
NPOPULATION = 10
MAX_ITERATION = 500
# defines CMA-ES algorithm solver
cmaes = CMAES(NPARAMS,
              popsize=NPOPULATION,
              weight_decay=0.0,
              sigma_init = 0.5
          )


def train(solver, env, policy):
    logs = []
    for j in range(MAX_ITERATION):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            policy.set_flat_weight(solutions[i])
            fitness_list[i] = rollout(env, policy)
        solver.tell(fitness_list)
        result = solver.result() # first element is the best solution, second element is the best fitness
        logs.append(result[1])
        if (j+1) % 100 == 0:
            print("fitness at iteration", (j+1), result[1])
    print("Local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])
    
    return logs, result[0]
logs, best_params = train(cmaes, env, cpg_rbfn)   


def plot_signal(cpg_rbfn, best_params):
    cpg_rbfn.set_flat_weight(best_params)
    cpg_rbfn.reset()
    output = np.array([cpg_rbfn.get_action() for _ in range(200)])

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    x = np.arange(output.shape[0])
    for i in range(3):
        y =  output[:, i]
        ax.plot(x, y)
plt.show()
plot_signal(cpg_rbfn, best_params)


# +
def render_path(env, cpg_rbfn, best_params):
    cpg_rbfn.set_flat_weight(best_params)
    obs = env.reset()
    cpg_rbfn.reset()
    path_return = 0.
    path_length = 0
    # fps = env._gym_env._gym_env.metadata['video.frames_per_second']
    fps = 100
    image_obss = []
    for step in tqdm(range(100)):
        act = cpg_rbfn.get_action(obs)
        act[0] = 0
        total_act = np.concatenate([act, act, -act, -act])
        next_obs, rew, done, info = env.step(total_act)
        image_obs = env.render(mode='rgb_array')
        image_obss.append(image_obs)
        path_return += rew
        path_length += 1
        if done: break
    print(path_length, path_return)
    image_obss = np.array(image_obss)
    write_gif('test_rendered_path', image_obss, fps)
    
render_path(env, cpg_rbfn, best_params)
# -




