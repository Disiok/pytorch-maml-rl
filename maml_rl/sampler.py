import gym
import torch
import multiprocessing as mp

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env


class TrajectorySampler(object):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
    
    def sample(self, policy, device='cpu'):
        observation = self.env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                observation_tensor = torch.from_numpy(observation).to(device=device)
                action_tensor = policy(observation_tensor).sample()
                action = action_tensor.cpu().numpy()
            new_observation, reward, done, info = self.env.step(action)
            self.env.render(mode='human')
            observation = new_observation

    def reset_task(self, task):
        self.env.unwrapped.reset_task(task)
        return True

    def sample_tasks(self, num_tasks):
        tasks = self.env.unwrapped.sample_tasks(num_tasks)
        return tasks


class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        self._env = gym.make(env_name)

    def sample(self, policy, params=None, gamma=0.95, device='cpu', delay=20, intrinsic=None):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device, delay=delay)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()

                if intrinsic is not None:
                    intrinsic_rewards_tensor = intrinsic(torch.cat([observations_tensor, actions_tensor], dim=1))
                    intrinsic_rewards = intrinsic_rewards_tensor.cpu().numpy()
                else:
                    intrinsic_rewards = None
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids, intrinsic_rewards=intrinsic_rewards)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
