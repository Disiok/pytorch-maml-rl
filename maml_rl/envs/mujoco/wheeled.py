from IPython import embed
import random
import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class WheeledEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """From MAESN project
    """
    def __init__(self, model_path='wheeled_maesn.xml'):
        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, fullpath, 5)

    def step(self, action):
        """Only task specific implementation is provided
        """
        raise NotImplementedError

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer(mode).render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            return data
        elif mode == 'human':
            self._get_viewer(mode).render()

    def reset_model(self):
        """Only task specific implementation is provided
        """
        raise NotImplementedError

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten()


class WheeledTaskEnv(WheeledEnv):
    """Pusher environment with multiple tasks and sparse rewards:
    (https://github.com/RussellM2020/maesn_suite/blob/master/maesn/rllab/envs/mujoco/pusher.py)
    """
    def __init__(self, task={}, sparse=False):
        self._task = task
        self._goal_pos = task.get('position', np.zeros((2,), dtype=np.float32))
        self._action_scaling = None
        self._sparse = sparse
        super(WheeledTaskEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        goal_distance = np.linalg.norm(observation[:2] - self._goal_pos)
        goal_position = np.linalg.norm(self._goal_pos)

        if self._sparse and goal_distance > 0.8 :
            goal_reward = -goal_position
        else:
            goal_reward = -goal_distance
        reward = goal_reward - ctrl_cost

        done = False
        infos = dict(reward_goal=goal_reward, reward_ctrl=-ctrl_cost, reward_total=reward)

        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks, seed=None):
        if seed is not None:
            np.random.seed(seed)

        radius = 2.0
        angle = np.random.uniform(0, np.pi, size=(num_tasks,))
        xpos = radius * np.cos(angle)
        ypos = radius * np.sin(angle)
        positions = np.concatenate([xpos[:, None], ypos[:, None]], axis=1)
        tasks = [{'task_id': task_id, 'position': position} for task_id, position in enumerate(positions)]
        return tasks

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        self.model.body_pos[-1][:2] = self._goal_pos
        return self._get_obs()

    def reset_task(self, task):
        self._task = task
        self._goal_pos = task['position']