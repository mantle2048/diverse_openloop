import gym
import torch
import ray
import numpy as np
import random
import logging
from copy import deepcopy
from typing import Dict, Callable
from collections import defaultdict

from reRLs.infrastructure.utils.utils import Path, get_pathlength
from open_loop.envs.make_env import make_env
from open_loop.envs.wrappers.trajectory_generator_wrapper_env import TrajectoryGeneratorWrapperEnv

ROLLOUT_LEN = 200

class Worker():

    def __init__(self, env, worker_id, config: Dict):
        self.worker_id = worker_id
        self.env = deepcopy(env)
        self.config = config
        self.n_path = config['n_path']
        np.random.seed(config['seed'] + worker_id)
        self.env.seed(config['seed'] + worker_id)
        random.seed(config['seed'] + worker_id)
        torch.manual_seed(config['seed'] + worker_id)

    def sample(self, traj = None, render=False) -> Dict:

        paths = rollouts(self.n_path, self.env, traj, render)
        ep_len = np.mean([get_pathlength(path) for path in paths])
        ep_rew = np.mean([path["rew"].sum() for path in paths])
        return (ep_len, ep_rew)

    def set_weight(self, weight):
        self.env.trajectory_generator.set_flat_weight(weight)

    @classmethod
    def as_remote(cls):
        return ray.remote(cls)

    def close(self):
        self.env.close()


class WorkerSet():

    def __init__(self, num_workers, env, config: Dict):
        self.num_workers = num_workers
        self.config = config
        self.env = env

        self.local_worker = self._make_worker(
            cls = Worker,
            env = self.env,
            worker_id = 0,
            config = self.config
        ) 
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.remote_workers = []
        self._add_workers(num_workers)

    def sync_weights(self, worker_weights):
        assert len(worker_weights) == self.num_workers, "num weight must equal num worker"
        for to_worker, weight in zip(self.remote_workers, worker_weights):
            to_worker.set_weight.remote(weight)

    def close(self):
        self.env.close()
        try:
            self.local_worker.close()
            tids = [w.close.remote() for w in self.remote_workers]
            ray.get(tids)
        except Exception:
            logging.warning("Failed to stop workers!")
        finally:
            for w in self.remote_workers:
                w.__ray_terminate__.remote()

    def _add_workers(self, num_workers):
        old_num_workers = len(self.remote_workers)
        self.remote_workers.extend(
            [
                self._make_worker(
                    cls = Worker.as_remote().remote,
                    env = self.env,
                    worker_id = old_num_workers + i + 1,
                    config = self.config
                )
                for i in range(num_workers)
            ]
        )

    def _make_worker(self, cls: Callable, env, worker_id, config: Dict):
        worker = cls(env, worker_id, config)
        return worker

def rollout(env, traj=None, render=False):
    if traj is None:
        assert isinstance(env, TrajectoryGeneratorWrapperEnv), \
            "must provide action sqeuence for env without traj generator"

    obs = env.reset()
    obss, acts, rews, next_obss, terminals, image_obss = [], [], [], [], [], []

    for step in range(ROLLOUT_LEN):
        if render:
            if hasattr(env, 'sim'):
                image_obss.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
            else:
                image_obss.append(env.render(mode='rbg_array'))
        obss.append(obs)

        # act is dummy for traj generator env
        if traj is not None:
            act = traj[step % len(traj)].clip(-1, 1)
        else:
            act = env.action_space.sample()
        acts.append(act)

        next_obs, rew, done, _ = env.step(act)

        rews.append(rew)
        next_obss.append(next_obs)

        rollout_done = done
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obss, image_obss, acts, rews, next_obss, terminals)

def rollouts(n_path, *args, **kwargs):
    paths = [rollout(*args, **kwargs) for _ in range(n_path)]
    return paths

def traj_rollouts(trajs, env, render=False):
    paths = [rollout(env, traj, render) for traj in trajs]
    return paths

def serial_sample(worker_set: WorkerSet, trajs = None, render = False):

    n_path = len(worker_set.remote_workers)
    worker = worker_set.local_worker

    if trajs is None:
        logs = [ worker.sample(render=render) for _ in range(n_path) ]
    else:
        assert len(trajs) == n_path, "num traj must equal num workers"
        logs = [ worker.sample(traj=traj, render=render) for traj in trajs ]

    log_dict = defaultdict(list)
    for (ep_len, ep_rew) in logs:
        log_dict['ep_lens'].append(ep_len)
        log_dict['ep_rews'].append(ep_rew)
    return log_dict

def parallel_sample(worker_set: WorkerSet, trajs = None, render = False):

    if trajs is None:
        logs = ray.get(
            [ worker.sample.remote(render=render) for worker in worker_set.remote_workers ]
        )
    else:
        logs = ray.get(
            [ worker.sample.remote(traj=traj, render=render) \
             for worker, traj in zip(worker_set.remote_workers, trajs) ]
        )

    log_dict = defaultdict(list)
    for (ep_len, ep_rew) in logs:
        log_dict['ep_lens'].append(ep_len)
        log_dict['ep_rews'].append(ep_rew)
    return log_dict






