import maml_rl.envs
import gym
import numpy as np
import torch
import json
import logging
from IPython import embed

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy, IntrinsicReward
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import TrajectorySampler

from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main(args):
    continuous_actions = (args.env_name in ['Wheeled-v0', 'Pusher-v0', 'AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'AntGoalRing-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0'])

    # Create folder for video renders
    render_folder = './renders/{0}'.format(args.output_folder)
    if not os.path.exists(render_folder):
        os.makedirs(render_folder)

    sampler = TrajectorySampler(args.env_name)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.env.observation_space.shape)),
            int(np.prod(sampler.env.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.env.observation_space.shape)),
            sampler.env.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)

    if args.policy_checkpoint is not None:
        logger.debug('Using policy checkpoint from: {}'.format(args.policy_checkpoint))
        state_dict = torch.load(args.policy_checkpoint)
        policy.load_state_dict(state_dict)

    task = sampler.sample_tasks(num_tasks=5, seed=999)[0]
    sampler.reset_task(task)
    episodes = sampler.sample(policy)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')
    
    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--policy-checkpoint', type=str,
        help='path for the policy checkpoint')
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    args = parser.parse_args()

    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
