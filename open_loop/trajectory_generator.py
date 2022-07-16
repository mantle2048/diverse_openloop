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
import matplotlib.pyplot as plt

from gym import spaces
from torch import nn
from itertools import cycle
from typing import Any, Callable, List, Dict, Optional, Union

from motion_imitation.robots import laikago_pose_utils

from reRLs.infrastructure.utils import pytorch_util as ptu

def init_weights(m: nn.Module, gain: float = 1):
    """
        Orthogonal initialization (used in PPO and A2C)
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # torch.nn.init.xavier_normal_(m.weight, gain=gain)
        torch.nn.init.normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.00)

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
        self.reset()

    def reset(self):
        pass

    def get_action(self, t) -> np.ndarray:
        return self._sines(t)

    def plot_curve(self, ax):
        x = np.arange(0, (self.period * 3) + self.timestep, self.timestep)
        y = self.get_action(x)
        ax.plot(x, y[:, 0], linewidth=3)
        ax.plot(x, y[:, 1], linewidth=3)
        return ax

    def _sines(self, t):
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
        self._init_weight(cpg_net)
        self.reset()

    def reset(self):
        ''' init weigth and period_signal '''
        pass

    def get_action(self, x) -> np.ndarray:
        ''' x is the cpg signal '''
        x = np.tile(x[:,None], (1, self.num_rbf, 1))
        c = np.tile(self.centres, (x.shape[0], 1, 1))
        distances = np.sqrt(np.square(x - c).sum(-1)) / np.exp(self.log_sigmas)
        return self.kernel_func(distances).squeeze()

    def plot_curve(self, ax, cpg_net, join=True):
        t = np.arange(0, cpg_net.period * 3 + cpg_net.timestep, cpg_net.timestep)
        x = cpg_net.get_action(t)
        y = self.get_action(x)
        if join:
            for i in range(self.num_rbf):
                ax.plot(t, y[:, i], linewidth=3)
        else:
            for i, a in enumerate(ax):
                a.plot(t, y[:, i], linewidth=3)
        return ax

    def _init_weight(self, cpg_net):
        # set rbf centres "u_i^{a_j} = a_j(T*(i-1)/(M-1)); i=1,2,...,M, j=0,1"
        t = np.linspace(
            0, cpg_net.period,
            self.num_rbf, endpoint=False
        )
        self.centres = cpg_net.get_action(t)
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
        self.num_rbf = num_rbf
        self.num_act = num_act

        self.cpg = CentralPatternGeneratorNetwork(sin_config, timestep)
        self.rbf = RadialBasisFunctionNetwork(
            num_rbf, cpg_net=self.cpg
        )
        self.linear = nn.Linear(num_rbf, num_act)
        self.apply(init_weights)

        self.timestep = self.cpg.timestep
        self.period = self.cpg.period

        self._init_weight()
        self.reset()

    def reset(self):
        self.cpg.reset()
        self.rbf.reset()

    def get_action(self, t) -> np.ndarray:
        x = self.cpg.get_action(t)
        raw_action = self.rbf.get_action(x)
        action = self.linear(ptu.from_numpy(raw_action))
        action *= 3
        return ptu.to_numpy(action)

    def get_observation(self, input_observation):
        """Get the trajectory generator's observation."""
        return input_observation

    def plot_curve(self, ax):
        x = np.arange(0, self.period * 3 + self.timestep, self.timestep)
        y = self.get_action(x)
        for i in range(self.num_act):
            ax.plot(x, y[:, i], linewidth=3)
        return ax

    def get_flat_weight(self):
        return ptu.to_numpy(self.linear.weight.data.flatten())

    def set_flat_weight(self, params):
        self.linear.weight.data = ptu.from_numpy(params).reshape(self.num_act, self.num_rbf)

    def get_state(self):
        return {k: v.cpu().detach() for k, v in self.state_dict().items()}

    def _init_weight(self):
        pass

    @property
    def num_params(self):
        return self.num_rbf * self.num_act

