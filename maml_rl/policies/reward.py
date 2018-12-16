import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from collections import OrderedDict
from maml_rl.policies.policy import weight_init


class IntrinsicReward(nn.Module):
    """Intrinsic reward network based on a multi-layer perceptron (MLP) with a scalar output
    """
    def __init__(self,
                 input_size,
                 hidden_sizes=(),
                 nonlinearity=torch.tanh,
                 init_std=1.0,
                 min_std=1e-6,
                 reward_importance=1.0):
        super(IntrinsicReward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.reward = nn.Linear(layer_sizes[-1], 1)

        self.reward_importance = torch.nn.Parameter(torch.Tensor(1))
        torch.nn.init.constant_(self.reward_importance, reward_importance)

        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                weight=params['layer{0}.weight'.format(i)],
                bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        reward = F.linear(output, weight=params['reward.weight'],
            bias=params['reward.bias'])
        return F.tanh(reward) * self.reward_importance

    def update_params(self, loss, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        grads = torch.autograd.grad(loss, self.parameters(),
            create_graph=not first_order)
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            if name == 'reward_importance':
                continue

            updated_params[name] = param - step_size * grad

        return updated_params
