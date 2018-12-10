#
#
#

import torch
import numpy as np

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init


class MAESNNormalMLPPolicy(Policy):
    """
    MAESN Normal MLP Policy.
    """

    def __init__(self,
                 observation_dim,
                 latent_dim,
                 action_dim,
                 hidden_dims,
                 num_tasks,
                 init_std=1.0,
                 min_std=1e-6,
                 default_step_size=0.5,
                 nonlinearity=torch.nn.functional.relu):
        """
        Initialization.

        @required
        :param observation_dim [int]:        Dimension of observation space.
        :param latent_dim      [int]:        Dimension of latent space.
        :param action_dim      [int]:        Dimension of the action space.
        :param hidden_dims     [tuple<int>]: Dimensions of the hidden layers.
        :param num_tasks       [int]:        Number of tasks.

        @optional
        :param init_std        [float]:               Initial std of normal policy.
        :param min_std         [float]:               Minimum std of normal policy.
        :param nonlinearity    [torch.nn.functional]: Nonlinearity functional.


        """
        super(MAESNNormalMLPPolicy, self).__init__(
            input_size=observation_dim + latent_dim,
            output_size=action_dim
        )

        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self.num_tasks = num_tasks
        self.min_log_std = np.log(min_std)
        self.nonlinearity = nonlinearity

        self.input_dim = self.observation_dim + self.latent_dim
        self.output_dim = self.action_dim

        layer_dims = (self.input_dim,) + self.hidden_dims
        self.num_layers = len(layer_dims)

        for i in range(1, self.num_layers):
            self.add_module(
                'layer{0}'.format(i),
                torch.nn.Linear(layer_dims[i - 1], layer_dims[i])
            )

        self.mu = torch.nn.Linear(layer_dims[-1], self.output_dim)
        self.sigma = torch.nn.Parameter(torch.Tensor(self.output_dim))

        self.latent_mus = torch.nn.Parameter(torch.Tensor(self.num_tasks, self.latent_dim))
        self.latent_sigmas = torch.nn.Parameter(torch.Tensor(self.num_tasks, self.latent_dim))

        torch.nn.init.constant_(self.sigma, np.log(init_std))
        torch.nn.init.normal_(self.latent_mus)
        torch.nn.init.constant_(self.latent_sigmas, 0)
        self.apply(weight_init)

        self.latent_mus_step_size = torch.nn.Parameter(torch.Tensor(1, self.latent_dim))
        self.latent_sigmas_step_size = torch.nn.Parameter(torch.Tensor(1, self.latent_dim))

        torch.nn.init.constant_(self.latent_mus_step_size, default_step_size)
        torch.nn.init.constant_(self.latent_sigmas_step_size, default_step_size)

    #def latent_distribution(self, task_id):
    #    """

    #    """
    #    mu = self.latent_mus[task_id]
    #    sigma = torch.exp(self.latent_sigmas[task_id])
    #    return torch.distributions.Normal(mu, sigma)

    #def latent_prior_distribution(self):
    #    """

    #    """
    #    zeros = torch.zeros_like(self.latent_mus[0])
    #    ones = torch.ones_like(self.latent_sigmas[0])
    #    return torch.distributions.Normal(zeros, ones)

    #def forward(self, observations, latent_noise, task_ids, params=None):
    def forward(self, observations, noise, task_ids, params=None):
        """
        Perform a forward pass of MAESNNormalMLPPolicy.

        :param observations [torch.FloatTensor]:  A [H x N x C_obs] observation tensor.
        :param latent_noise [torch.FloatTensor]:  A [N x C_latent] noise tensor.
        :param task_ids     [torch.LongTensor]:   A [N] index tensor of task ids.
        :param params       [OrderedDict]:
        :return             [torch.distribution]: A [H x N x C_act] Normal distribution.
        """
        # Sanity checks.
        #assert(observations.dim() == 3)
        #assert(latent_noise.dim() == 2)
        #assert(task_ids.dim() == 1)

        # Retrieve current parameters if necessary.
        params = OrderedDict(self.named_parameters()) if params is None else params

        # Sample noise from relevant latent spaces.
        latent_mus = torch.index_select(params['latent_mus'], 0, task_ids)        # [N x C_latent]
        latent_sigmas = torch.index_select(params['latent_sigmas'], 0, task_ids)  # [N x C_latent]
        zs = latent_mus + noise * torch.exp(latent_sigmas)


        # Construct noise-agumented input features.
        #latent_zs = latent_noise.unsqueeze(0).expand(observations.size(0), -1, -1)
        #output = torch.cat([observations, latent_zs], dim=-1)
        #zs = noise if observations.dim() == 2 else noise.unsqueeze(0).expand(observations.size(0), -1, -1)
        zs = zs if observations.dim() == 2 else zs.unsqueeze(0).expand(observations.size(0), -1, -1)
        output = torch.cat([observations, zs], dim=-1)

        # Pass through MLP.
        for i in range(1, self.num_layers):
            output = torch.nn.functional.linear(
                output,
                weight=params['layer{0}.weight'.format(i)],
                bias=params['layer{0}.bias'.format(i)]
            )
            output = self.nonlinearity(output)

        # Extract action distribution mean.
        mu = torch.nn.functional.linear(
            output,
            weight=params['mu.weight'],
            bias=params['mu.bias']
        )

        # Extract action distribution sigma.
        sigma = torch.exp(torch.clamp(
            params['sigma'],
            min=self.min_log_std
        ))

        if observations.dim() == 2:
            sigma = sigma.view(1, -1).expand(observations.size(0), -1)
        else:
            sigma = sigma.view(1, 1, -1).expand(observations.size(0), observations.size(1), -1)

        return torch.distributions.Normal(mu, sigma)

    def update_params(self,
                      loss,
                      step_size=0.5,
                      first_order=False,
                      latent_only=True):
        """
        Apply one-step gradient update on loss function.

        @required
        :param loss        [torch.Tensor]: The loss value.

        @optional
        :param step_size   [float]:        The default step size.
        :param first_order [bool]:         Use first order approximation.
        :param latent_only [bool]:         Only optimize latent space.

        :return            [OrderedDict]:  The updated parameters.
        """
        grad_params = []
        named_grad_params = []
        for (name, param) in self.named_parameters():
            if (name == 'latent_mus_step_size' or
                name == 'latent_sigmas_step_size'):
                continue

            if not latent_only:
                grad_params.append(param)
                named_grad_params.append((name, param))
            elif (name == 'latent_mus' or name == 'latent_sigmas'):
                grad_params.append(param)
                named_grad_params.append((name, param))

        grads = torch.autograd.grad(
            loss,
            grad_params,
            create_graph=not first_order
        )

        step_sizes = {name: step_size for (name, _) in self.named_parameters()}
        step_sizes['latent_mus'] = self.latent_mus_step_size
        step_sizes['latent_sigmas'] = self.latent_sigmas_step_size

        updated_params = OrderedDict()
        for (name, param) in self.named_parameters():
            updated_params[name] = param

        for (name, param), grad in zip(named_grad_params, grads):
            updated_params[name] = param - step_sizes[name] * grad

        return updated_params



    #def update_params(self, loss, step_size,
    #                  latent_only=True, first_order=False):
    #    """

    #    :param loss              [torch.Tensor]: The loss.
    #    :param default_step_size [float]:        The default step size.
    #    :param latent_only       [bool]:         Only update latent parameters.
    #    :param first_order       [bool]:         First order approximation of gradients.

    #    TODO(kwong): Add adaptive step-sizes for latent variables.
    #    TODO(kwong): Add ability to turn off theta updates.
    #    NOTE(kwong): Step sizes are not adapted here.
    #    """
    #    params, params_keys = [], []
    #    latent_params, latent_params_keys = [], []
    #    for (name, param) in self.named_parameters():
    #        if name in ['latent_mus', 'latent_sigmas']:
    #            latent_params.append(param)
    #            latent_params_keys.append(name)
    #        elif name not in ['latent_mus_step_size', 'latent_sigmas_step_size']:
    #            params.append(param)
    #            params_keys.append(name)

    #    print(params_keys)
    #    grads = torch.autograd.grad(
    #        loss,
    #        params,#grad_params,#self.parameters(),
    #        create_graph=not first_order
    #    )

    #    updated_params = OrderedDict()
    #    #for (name, param) in self.named_parameters():
    #    #    updated_params[name] = param

    #    for name, param, grad in zip(params_keys, params, grads):
    #        updated_params[name] = param - step_size * grad

    #    #updated_params['latent_mus'] = grad_params[0] - self.latent_mus_step_size * grads[0]
    #    #updated_params['latent_sigmas'] = grad_params[1] - self.latent_sigmas_step_size * grads[1]

    #    #updated_params = OrderedDict()
    #    #for (name, param), grad in zip(self.named_parameters(), grads):
    #    #    updated_params[name] = param - step_size * grad

    #    return updated_params