import numpy as np
import torch
import reRLs.infrastructure.utils.pytorch_util as ptu

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from typing import List, Dict, Tuple, Union
from torch.distributions import Normal, MultivariateNormal

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

def init_weight(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.00)

def build_mlp(
        input_size: int,
        output_size: int,
        layers: List = [256, 256],
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        with_batch_norm = False
):
    in_size = input_size
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    mlp_layers = []
    for size in layers:
        mlp_layers.append(nn.Linear(in_size, size))
        if with_batch_norm:
            mlp_layers.append(nn.BatchNorm1d(size))
        mlp_layers.append(activation)
        in_size = size
    mlp_layers.append(nn.Linear(in_size, output_size))
    mlp_layers.append(output_activation)
    return nn.Sequential(*mlp_layers)

class VAE(nn.Module):

    def __init__(
        self,
        input_size,
        variable_size,
        hidden_layers,
        learning_rate = 0.001,
    ):

        super().__init__()

        self.input_size = input_size # [ num_point, act_dim ]
        self.variable_size = variable_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        self.encoder_net = build_mlp(
            np.prod(input_size),
            variable_size,
            layers=hidden_layers,
            activation='relu',
            with_batch_norm=True,
        )

        log_std = -0.5 * torch.ones(variable_size, dtype=torch.float32)
        self.log_std = nn.Parameter(log_std)

        self.decoder_net = build_mlp(
            variable_size,
            np.prod(input_size),
            layers=hidden_layers,
            activation='relu',
            with_batch_norm=True,
        )

        self.MSELoss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.99)

        self.apply(init_weight)

    def encoder(self, x: torch.Tensor):

        if len(x.shape) == 1:
            x = x[None]
        mu = self.encoder_net(x)
        std = torch.exp(self.log_std)

        return mu, std

    def sample(self, mu: torch.Tensor, std: torch.Tensor):

        batch_size = mu.shape[0]
        epsilon = torch.randn(batch_size, self.variable_size)
        z = mu + torch.mul(epsilon, std)

        return z

    def decoder(self, z: torch.Tensor):

        x = self.decoder_net(z)

        return x

    def forward(self, x: np.ndarray):

        x = ptu.from_numpy(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        mu, std = self.encoder(x)
        z = self.sample(mu, std)
        reconstruct_x = self.decoder(z)

        return reconstruct_x, mu, std, np.mean(z.detach().numpy().flatten())

    def update(self, x: np.ndarray, rew_list: List):
        '''
        p = N(mu1,sigma1), q = N(mu2, sigma2)
        KL(p||q) = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / 2*sigma2^2 - 1/2
        '''
        self.train()
        rews = torch.as_tensor(rew_list)
        normalized_rews = (rews - rews.mean()) / rews.std()
        exp_rews = torch.exp(normalized_rews)

        reconstruct_x, mu, std, z_mean = self.forward(x)

        x = ptu.from_numpy(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # reconstruction_loss = 0.5 * torch.sum((reconstruct_x - x) ** 2, dim=-1)
        # kl_divergence = -0.5 * torch.sum(1 + 2 * std.log() - mu ** 2 - std ** 2, dim = 1)
        reconstruction_loss = self.MSELoss(reconstruct_x, x)
        kl_divergence = torch.mean(-0.5 * torch.sum(1 + 2 * std.log() - mu ** 2 - std ** 2, dim = 1), dim = 0)

        # elbo_loss = torch.mean((reconstruction_loss + kl_divergence) * (-exp_rews))
        elbo_loss = torch.mean((reconstruction_loss + kl_divergence))

        self.optimizer.zero_grad()
        elbo_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        train_log = {}
        train_log['elbo_loss'] = elbo_loss.item()
        train_log['reconstruction_loss'] = torch.mean(reconstruction_loss).item()
        train_log['kl_divergence'] = torch.mean(kl_divergence).item()
        train_log['z_mean'] = z_mean

        return train_log

    def generate(self, z: np.ndarray):

        self.eval()
        z = ptu.from_numpy(z)
        if len(z.shape) > 1:
            batch_size = z.shape[0]
        else:
            batch_size = 1
        z = z.view(batch_size, -1)

        with torch.no_grad():
            reconstruct_x = self.decoder(z)

        reconstruct_x = np.array([ i.view(*self.input_size).numpy() for i in reconstruct_x ])
        return reconstruct_x

    def set_state(self, state_dict):
        self.load_state_dict(state_dict)

    def get_state(self):
        return {k: v.cpu().detach() for k, v in self.state_dict().items()}
