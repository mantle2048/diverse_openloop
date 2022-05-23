from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import attr
import numpy as np
import torch

from gym import spaces
from torch import nn
from itertools import cycle
from typing import Any, Callable, List, Dict, Optional, Union

from motion_imitation.robots import laikago_pose_utils

from reRLs.infrastructure.utils import pytorch_util as ptu

class CentralPatternGeneratorNetwork():

    def __init__(
        self,
        sine_config: Dict=None,
        timestep: float=0.01,
    ):
        self.amplitude = sine_config['amplitude']
        self.theta = sine_config['theta']
        self.frequency = sine_config['frequency']

        self.timestep = timestep

        self.period = 1 / sine_config['frequency']

        # remember to add 1 for total timestep
        self.timestep_per_period = int(self.period / self.timestep)
        self.reset()

    def reset(self):
        pass


    def get_action(self, t) -> np.ndarray:
        return self._sines(t)

    def _sines(t):
        phase1 =  2 * np.pi * self.frequency * t
        phase2 =  2 * np.pi * self.frequency * t + self.theta
        phase = np.vstack([phase1, phase2]).T
        return self.amplitude * np.sin(phase)

# RBFs

def gaussian(alpha):
    phi = np.exp(-1*np.square(alpha))
    return phi

class RadialBasisFunctionNetwork():

    def __init__(
        self,
        num_rbf: int,
        cpg_net: CentralPatternGeneratorNetwork,
        kernel_func: Callable=gaussian
    ):
        self.num_rbf = num_rbf
        self.kernel_func = kernel_func
        self.cpg_net = cpg_net
        self._init_weight()
        self.reset()

    def reset(self):
        ''' init weigth and period_signal '''

        self.period_signal = self._get_period_signal(self.cpg_net)
        self.cycle_signal = cycle(self.period_signal)

    def get_action(self, obs: np.ndarray=None) -> np.ndarray:
        # obs is dummy for rbf, just for consistent api.
        return next(self.cycle_signal)

    def _get_period_signal(self, cpg_net):

        x = np.tile(cpg_net.period_signal[:, None], (1, self.num_rbf, 1))
        c = np.tile(self.centres[None], (cpg_net.timestep_per_period, 1, 1))
        distances = np.sqrt(np.square(x - c).sum(-1)) / np.exp(self.log_sigmas)
        return self.kernel_func(distances).squeeze()

    def _init_weight(self):
        # set rbf centres "u_i^{a_j} = a_j(T*(i-1)/(M-1)); i=1,2,...,M, j=0,1"
        idx = np.linspace(
            0, self.cpg_net.timestep_per_period,
            self.num_rbf, endpoint=False, dtype=np.int64
        )
        self.centres = self.cpg_net.period_signal[idx]
        self.log_sigmas = np.zeros(self.num_rbf) + np.log(0.5)

    def _set_cpg_net(self, cpg_net):
        self.cpg_net = cpg_net

class CpgRbfNet(nn.Module):

    def __init__(
        self,
        sin_config: Dict,
        timestep: int,
        num_rbf: int,
        num_act: int
    ):
        super().__init__()
        self.sin_config = sin_config
        self.timestep = timestep

        self.num_rbf = num_rbf
        self.num_act = num_act

        self.cpg = CentralPatternGeneratorNetwork(sin_config, timestep)
        self.rbf = RadialBasisFunctionNetwork(
            num_rbf, cpg_net=self.cpg
        )
        self.linear = nn.Linear(num_rbf, num_act)
        self.timestep_per_period = self.cpg.timestep_per_period
        self._init_weight()
        self.reset()

    def reset(self):

        self.cpg.reset()
        self.rbf.reset()

        self.period_signal = self._get_period_signal()
        self.cycle_signal = cycle(self.period_signal)

    def get_action(self, obs: np.ndarray=None) -> np.ndarray:
        act = next(self.cycle_signal) - self.period_signal[0]
        return ptu.to_numpy(act)

    def get_flat_weight(self):
        return ptu.to_numpy(self.linear.weight.data.flatten())

    def set_flat_weight(self, params):
        self.linear.weight.data = ptu.from_numpy(params).reshape(self.num_act, self.num_rbf)

    def get_state(self):
        return {k: v.cpu().detach() for k, v in self.state_dict().items()}

    def _init_weight(self):
        pass

    def _get_period_signal(self):
        return self.linear(ptu.from_numpy(self.rbf.period_signal))

    @property
    def num_params(self):
        return self.num_rbf * self.num_act

