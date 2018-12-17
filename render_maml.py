#
#
#

import maml_rl.envs

import gym
import json
import time
import torch
import random
import logging
import numpy as np

from maml_rl.envs import CONTINUOUS_ENVS
from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler, TrajectorySampler
from maml_rl.utils import task_utils, torch_utils

from torch.nn.utils.convert_parameters import (
    vector_to_parameters, parameters_to_vector
)

from tensorboardX import SummaryWriter


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(args):
    """

    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    continuous_actions = (args.env_name in CONTINUOUS_ENVS)
    writer_path = os.path.join(args.out, 'logs')
    save_folder = os.path.join(args.out, 'saves')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    assert(os.path.exists(args.tasks))
    task_distribution = task_utils.normalize_task_ids(torch.load(args.tasks))[:50]

    sampler = TrajectorySampler(args.env_name)

    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.env.observation_space.shape))
    )

    task_rewards = {}
    for i, task in enumerate(task_distribution):
        if continuous_actions:
            policy = NormalMLPPolicy(
                int(np.prod(sampler.env.observation_space.shape)),
                int(np.prod(sampler.env.action_space.shape)),
                (args.hidden_size,) * args.num_layers
            )

            if args.checkpoint is not None:
                assert(os.path.exists(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
                policy.load_state_dict(checkpoint)
        else:
            policy = CategoricalMLPPolicy(
                int(np.prod(sampler.env.observation_space.shape)),
                sampler.env.action_space.n,
                hidden_sizes=(args.hidden_size,) * args.num_layers
            )
            if args.checkpoint is not None:
                assert(os.path.exists(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
                policy.load_state_dict(checkpoint)

        start = time.time()
        sampler.reset_task(task)
        sampler.sample(
            policy,
            device=args.device
        )
        logger.debug('Finished sampling episodes in {:.3f} seconds.'.format(time.time() - start))


if __name__ == '__main__':
    """

    """
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description='Reinforcement Learning with Model-Agnostic Meta-Learning (MAML)'
    )

    # General
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')

    # Policy network (relu activation function)
    parser.add_argument('--checkpoint', type=str, default=None,
        help='model checkpoint')
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')
    parser.add_argument('--latent-dim', type=int, default=2,
        help='dimension of the latent space')

    # Task-specific
    parser.add_argument('--tasks', type=str, default=None,
        help='task distribution')
    parser.add_argument('--fast-batch-size', type=int, default=40,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=100,
        help='number of batches')

    # Miscellaneous
    parser.add_argument('--out', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()
    print(args)

    # Create logs and saves folder if they don't exist
    args.out = os.path.expanduser(args.out)
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')

    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
