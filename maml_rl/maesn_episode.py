#
#
#

import torch
import numpy as np

class MAESNBatchEpisodes(object):
    """
    Encapsulates a batch of MAESN episodes for a single task.
    """

    def __init__(self, batch_size, task_id, noise, gamma=0.95, device='cpu'):
        """
        Initialization.

        :param batch_size [int]:          The number of trajectories.
        :param task_id    [int]:          The task id of this batch.
        :param noise      [torch.Tensor]: A [N x C_latent] noise tensor.
        :param gamma      [float]:        The reward discount factor.
        :param device     [str]:          The current device.
        """
        self.batch_size = batch_size
        self.task_id = task_id
        self.noise = noise
        self.gamma = gamma
        self.device = device

        self._observations_list = [[] for _ in range(batch_size)]
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]
        self._mask_list = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._task_ids = None
        self._mask = None

    @property
    def observations(self):
        """
        Return a tensor of observations.

        :return [torch.Tensor]: A [H x N x C_obs] feature tensor.
        """
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size)
                + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._observations_list[i])
                observations[:length, i] = np.stack(self._observations_list[i], axis=0)
            self._observations = torch.from_numpy(observations).to(self.device)
        return self._observations

    @property
    def actions(self):
        """
        Return a tensor of actions.

        :return [torch.Tensor]: A [H x N x C_act] feature tensor.
        """
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                actions[:length, i] = np.stack(self._actions_list[i], axis=0)
            self._actions = torch.from_numpy(actions).to(self.device)
        return self._actions

    @property
    def rewards(self):
        """
        Return a tensor of rewards.

        :return [torch.Tensor]: A [H x N] reward tensor.
        """
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rewards_list[i])
                rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
            self._rewards = torch.from_numpy(rewards).to(self.device)
        return self._rewards

    @property
    def returns(self):
        """
        Return a tensor of returns.

        :return [torch.Tensor]: A [H x N] return tensor.
        """
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            mask = self.mask.cpu().numpy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).to(self.device)
        return self._returns

    @property
    def task_ids(self):
        """
        Return a tensor of task ids.

        :return [torch.Tensor]: A [N]-dimension task id tensor.
        """
        if self._task_ids is None:
            self._task_ids = torch.tensor(
                [self.task_id] * self.batch_size,
                dtype=torch.long,
                device=self.device
            )

        return self._task_ids

    @property
    def mask(self):
        """
        Return a mask tensor of valid transitions.

        :return [torch.Tensor]: A [H x N] mask tensor.
        """
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                mask[:length, i] = 1.0
            self._mask = torch.from_numpy(mask).to(self.device)
        return self._mask

    def intrinsic_rewards(self, intrinsic):
        """

        """
        if intrinsic is None:
            return 0

        features = torch.cat([
            self.observations,
            self.actions,
        ], dim=2)

        intrinsic_rewards = intrinsic(features).squeeze(-1)
        return intrinsic_rewards

    def intrinsic_returns(self, intrinsic):
        """

        """
        if intrinsic is None:
            return 0

        returns = [None] * len(self)
        intrinsic_rewards = self.intrinsic_rewards(intrinsic)

        returns[-1] = intrinsic_rewards[-1] * self.mask[-1]
        for i in range(len(self) - 2, -1, -1):
            returns[i] = self.gamma * returns[i + 1] + intrinsic_rewards[i] * self.mask[i]

        intrinsic_returns = torch.stack(returns, dim=0)
        return intrinsic_returns

    def gae(self, values, intrinsic=None, tau=1.0):
        """
        Compute the Generalized Advantage Estimate.

        :param values    [torch.Tensor]:    A [H x N x 1] value tensor.
        :param intrinsic [torch.nn.Module]:
        :param tau       [float]:           The lambda/tau factor.
        :return          [torch.Tensor]:    A [H x N] advantage tensor.
        """
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = values.squeeze(2).detach()
        values = torch.nn.functional.pad(values * self.mask, (0, 0, 0, 1))

        mixed_rewards = self.rewards + self.intrinsic_rewards(intrinsic)
        deltas = mixed_rewards + self.gamma * values[1:] - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * tau + deltas[i]
            advantages[i] = gae

        return advantages

    def append(self, observations, actions, rewards, batch_ids):
        """
        Add a batch of transition tuples to the relevant episodes.

        :param observations [np.array]:   A [N' x C_obs] observation np.array.
        :param actions      [np.array]:   A [N' x C_act] action np.array.
        :param rewards      [np.array]:   A [N'] rewards np.array.
        :param batch_ids    [tuple<int>]: A list of batch ids.
        """
        zipped = zip(observations, actions, rewards, batch_ids)
        for observation, action, reward, batch_id in zipped:
            if batch_id is None:
                continue

            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(np.float32))
            self._rewards_list[batch_id].append(reward.astype(np.float32))

    def __len__(self):
        """
        Return the maximum length of an episode.

        :return [int]: The maximum length of an episode.
        """
        return max(map(len, self._rewards_list))