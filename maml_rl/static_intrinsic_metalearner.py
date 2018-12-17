#
#
#

import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (
    vector_to_parameters, parameters_to_vector
)

from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.torch_utils import (
    weighted_mean, detach_distribution, weighted_normalize
)


class StaticIntrinsicMetaLearner(object):
    """
    Intrinsic MetaLearner.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan,
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan,
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """

    def __init__(self,
                 sampler,
                 policy,
                 reward,
                 baseline,
                 gamma=0.95,
                 fast_lr=0.5,
                 tau=1.0,
                 device='cpu'):
        """
        Initialization.

        :param sampler  [MAESNBatchSampler]:
        :param policy   [MAESNPolicy]:
        :param reward   [Reward]:
        :param baseline [LinearFeatureBaseline]:
        :param gamma    [float]:
        :param fast_lr  [float]:
        :param tau      [float]:
        :param device   [str]:
        """
        self.sampler = sampler
        self.policy = policy
        self.reward = reward
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)

    def inner_loss(self,
                   episodes,
                   params=None,
                   use_intrinsic=True):
        """
        Compute the inner loss for a one-step gradient update.
        The inner loss is REINFORCE with baseline [2], computed on
        advantage estimates with Generalized Advantage Estimates [3].

        :param episodes      [MAESNBatchEpisode]:
        :param params        [OrderedDict]:
        """
        intrinsic = self.reward if use_intrinsic else None

        # Note: we assume that the baseline is already
        # fitted using extrinsic rewards from train_episodes.
        values = self.baseline(episodes)
        advantages = episodes.gae(
            values,
            intrinsic=intrinsic,
            tau=self.tau
        )
        advantages = weighted_normalize(
            advantages,
            weights=episodes.mask
        )

        pi = self.policy(
            episodes.observations,
            params=params
        )

        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)  # [H x N x C_act]

        loss = -weighted_mean(
            log_probs * advantages,
            dim=0,
            weights=episodes.mask
        )

        return loss

    def adapt_policy(self,
                     episodes,
                     first_order=False):
        """
        Adapt policy network using extrinsic + intrinsic rewards.

        :param episodes    [MAESNBatchEpisodes]:
        :param first_order [bool]:
        """
        # Fit the baseline to the training episodes.
        # Note: We only use extrinsic rewards here.
        self.baseline.fit(episodes)

        # Get the loss on the training episodes.
        loss = self.inner_loss(episodes, use_intrinsic=True)

        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(
            loss,
            step_size=self.fast_lr,
            first_order=first_order
        )

        return params

    def adapt_reward(self,
                     episodes,
                     policy_params=None,
                     first_order=False):
        """
        Adapt policy network using extrinsic rewards.

        """
        # Fit the baseline to the training episodes.
        # Note: We only use extrinsic rewards here.
        self.baseline.fit(episodes)

        # Get the loss on the training episodes.
        loss = self.inner_loss(
            episodes,
            params=policy_params,
            use_intrinsic=False
        )

        # Get the new parameters after a one-step gradient update
        params = self.reward.update_params(
            loss,
            step_size=self.fast_lr,
            first_order=first_order
        )

        return params


    def sample(self, tasks, first_order=False):
        """
        Sample trajectories before and after parameter updates.

        :param tasks       [list<dict>]:               A list of task descriptions.
        :param first_order [bool]:
        :return            [list<MAESNBatchEpisodes>]: A list of MAESNBatchEpisodes,
                                                       one for each task in `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)

            # Sample training trajectories.
            train_episodes = self.sampler.sample(
                self.policy,
                gamma=self.gamma,
                device=self.device
            )

            # Update parameters on training trajectories.
            params = self.adapt_policy(
                train_episodes,
                first_order=first_order
            )

            # Sample validation trajectories.
            valid_episodes = self.sampler.sample(
                self.policy,
                params=params,
                gamma=self.gamma,
                device=self.device
            )

            episodes.append((train_episodes, valid_episodes))

        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        """
        Compute the KL-divergence

        """
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt_policy(train_episodes)

            pi = self.policy(
                valid_episodes.observations,
                params=params
            )

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)

            kl = weighted_mean(
                kl_divergence(pi, old_pi),
                dim=0,
                weights=mask
            )
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, function, damping=1e-2):
        """
        Hessian-vector product, based on the Perlmutter method.

        :return [function]:
        """
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(
                kl,
                function.parameters(),
                create_graph=True
            )

            flat_grad_kl = parameters_to_vector(grads)
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, function.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def surrogate_loss(self,
                       episodes,
                       use_intrinsic,
                       old_pis=None):
        """
        Compute the surrogate loss used for TRPO.

        """
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            # Adapt the policy network using extrinsic + intrinsic rewards.
            params = self.adapt_policy(train_episodes)

            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(
                    valid_episodes.observations,
                    params=params
                )
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                intrinsic = self.reward if use_intrinsic else None

                # Note: we assume that the baseline is already
                # fitted using extrinsic rewards from train_episodes.
                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(
                    values,
                    intrinsic=intrinsic,
                    tau=self.tau
                )
                advantages = weighted_normalize(
                    advantages,
                    weights=valid_episodes.mask
                )

                log_ratio = (pi.log_prob(valid_episodes.actions)
                           - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(
                    ratio * advantages,
                    dim=0,
                    weights=valid_episodes.mask
                )
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)

                kl = weighted_mean(
                    kl_divergence(pi, old_pi),
                    dim=0,
                    weights=mask
                )
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def step(self,
             episodes,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        """
        Use TRPO to update policy and reward networks.


        """
        # Update policy network.
        policy_step = self.step_function(
            self.policy,
            episodes,
            max_kl=max_kl,
            cg_iters=cg_iters,
            cg_damping=cg_damping,
            ls_max_steps=ls_max_steps,
            ls_backtrack_ratio=ls_backtrack_ratio,
            use_intrinsic=True
        )

        # Update reward network.
        reward_step = self.step_function(
            self.reward,
            episodes,
            max_kl=max_kl,
            cg_iters=cg_iters,
            cg_damping=cg_damping,
            ls_max_steps=ls_max_steps,
            ls_backtrack_ratio=ls_backtrack_ratio,
            use_intrinsic=False
        )

        return policy_step, reward_step

    def step_function(self,
                      function,
                      episodes,
                      max_kl=1e-3,
                      cg_iters=10,
                      cg_damping=1e-2,
                      ls_max_steps=10,
                      ls_backtrack_ratio=0.5,
                      use_intrinsic=False):
        """
        Perform TRPO on some network.

        """
        old_loss, _, old_pis = self.surrogate_loss(
            episodes,
            use_intrinsic=use_intrinsic
        )

        # Compute gradients with respect to function.
        grads = torch.autograd.grad(old_loss, function.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hvp = self.hessian_vector_product(episodes, function, damping=cg_damping)
        stepdir = conjugate_gradient(hvp, grads, cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hvp(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(function.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(
                old_params - step_size * step,
                function.parameters()
            )

            loss, kl, _ = self.surrogate_loss(
                episodes,
                use_intrinsic=use_intrinsic,
                old_pis=old_pis
            )

            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break

            step_size *= ls_backtrack_ratio
        else:
            step_size = 0
            vector_to_parameters(old_params, function.parameters())

        return step_size * step

    def to(self, device, **kwargs):
        """

        """
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device
