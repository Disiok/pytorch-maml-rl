import random
import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """From MAESN project
    """
    def __init__(self, model_path='pusher_maesn.xml'):
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
            self.sim.data.qpos.flat[:3],
            self.sim.data.geom_xpos[-6:-1, :2].flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten()


class PusherTaskEnv(PusherEnv):
    """Pusher environment with multiple tasks and sparse rewards:
    (https://github.com/RussellM2020/maesn_suite/blob/master/maesn/rllab/envs/mujoco/pusher.py)
    """
    def __init__(self, task={}, sparse=False):
        self._task = task
        self._block_choice = task.get('block_choice', 0)
        self._goal = task.get('goal', np.array([0., 0.]))
        self._block_positions = task.get('block_positions', np.zeros((13, 1)))
        self._action_scaling = None
        self._sparse = sparse
        super(PusherTaskEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        curr_block_xidx = 3 + 2 * self._block_choice
        curr_block_yidx = 4 + 2 * self._block_choice

        curr_block_pos =  np.array([observation[curr_block_xidx], observation[curr_block_yidx]])

        block_dist = np.linalg.norm(self._goal - curr_block_pos)
        goal_dist = np.linalg.norm(self._goal)

        if self._sparse and block_dist > 0.2:
            reward = -5 * goal_dist
        else:
            reward = -5 * block_dist

        done = False
        infos = dict(block_distance=block_dist, goal_distance=goal_dist, block_choice=self._block_choice)

        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks, seed=None):
        if seed is not None:
            np.random.seed(seed)

        assert(num_tasks % 5 == 0)
        num_tasks = int(num_tasks / 5)

        # TODO(suo): Clean up this code
        tasks = []
        blockarr = np.array(range(5))
        for _ in range(num_tasks):
            blockpositions = np.zeros((13, 1))
            blockpositions[0:3] = 0
            positions_sofar = []
            np.random.shuffle(blockarr)
            for i in range(5):
                xpos = np.random.uniform(.35, .65)
                ypos = np.random.uniform(-.5 + 0.2*i, -.3 + 0.2*i)
                blocknum = blockarr[i]
                blockpositions[3+2*blocknum] = -0.2*(blocknum + 1) + xpos
                blockpositions[4+2*blocknum] = ypos

            for i in range(5):
                blockchoice = blockarr[i]
                goal_xpos = np.random.uniform(.75, .95)
                goal_ypos = np.random.uniform(-.5 + .2*i, -.3 + .2*i)
                curr_goal = {
                    'block_choice': blockchoice,
                    'goal': np.array([goal_xpos, goal_ypos]),
                    'block_positions': blockpositions[:, 0],
                }
                tasks.append(curr_goal)

        for task_id, task in enumerate(tasks):
            task['task_id'] = task_id

        random.shuffle(tasks)
        return tasks

    def reset_model(self):
        qpos = self._block_positions
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_task(self, task):
        self._task = task
        self._block_choice = task['block_choice']
        self._goal = task['goal']
        self._block_positions = task['block_positions']