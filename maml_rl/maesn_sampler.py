#
#
#

import gym
import torch
import multiprocessing as mp

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.maesn_episode import MAESNBatchEpisodes
from maml_rl.episode import BatchEpisodes


def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env


class MAESNBatchSampler(object):
    """
    Sample a batch of episodes for a single task.
    """

    def __init__(self,
                 env_name,
                 batch_size,
                 num_workers=mp.cpu_count() - 1):
        """
        Initialization.

        :param env_name    [str]: The environment name.
        :param batch_size  [int]: The number of trajectories.
        :param num_workers [int]: The number of workers.
        """
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.task_id = None
        self.queue = mp.Queue()

        self._env = gym.make(env_name)
        self.envs = SubprocVecEnv(
            [make_env(env_name) for _ in range(num_workers)],
            queue=self.queue
        )

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        """
        Sample a batch of episodes for the current task.

        :param policy [MAESNPolicy]:        The behaviour policy.
        :param params [OrderedDict]:        The policy params to use, or None.
        :param gamma  [float]:              The reward discount factor.
        :param device [str]:                The current device.
        :return       [MAESNBatchEpisodes]: A batch of MAESN episodes.
        """
        assert(self.task_id is not None)  # Task must be configured.

        # Sample Normal noise per episode.
        noise_distribution = torch.distributions.Normal(
            torch.zeros((self.batch_size, policy.latent_dim), device=device),
            torch.ones((self.batch_size, policy.latent_dim), device=device)
        )

        noise_tensor = noise_distribution.sample()
        task_tensor = torch.tensor([self.task_id], dtype=torch.long, device=device)

        # Construct a batch of episodes.
        episodes = MAESNBatchEpisodes(
            batch_size=self.batch_size,
            task_id=self.task_id,
            noise=noise_tensor,
            gamma=gamma,
            device=device
        )

        for i in range(self.batch_size):
            self.queue.put(i)

        for _ in range(self.num_workers):
            self.queue.put(None)

        dones = [False]
        observations, batch_ids = self.envs.reset()

        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                batch_ids_tensor = torch.tensor(
                    [bid if bid is not None else 0 for bid in batch_ids],
                    dtype=torch.long, device=device
                )

                current_noise_tensor = torch.index_select(noise_tensor, 0, batch_ids_tensor)
                current_task_tensor = task_tensor.expand(observations_tensor.size(0))

                actions_distribution = policy(
                    observations_tensor,#.unsqueeze(0),
                    current_noise_tensor,
                    current_task_tensor,
                    params=params
                )
                actions_tensor = actions_distribution.sample()
                #actions_tensor = actions_tensor.squeeze(0)
                actions = actions_tensor.cpu().numpy()

            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

        return episodes

    def reset_task(self, task):
        """
        Reset the sampler for the specified task.

        :param task [dict]: The task description.
        :return     [bool]: Success.
        """
        self.task_id = task.get('task_id', None)
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        """
        Sample a batch of tasks.

        :param num_tasks [int]:        The number of tasks to sample.
        :return          [list<dict>]: A list of task descriptions.
        """
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks