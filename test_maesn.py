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
from maml_rl.maesn_metalearner import MAESNMetaLearner
from maml_rl.policies import MAESNNormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.maesn_sampler import MAESNBatchSampler
from maml_rl.utils import torch_utils, task_utils

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
    task_distribution = task_utils.normalize_task_ids(torch.load(args.tasks))

    assert(os.path.exists(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)

    sampler = MAESNBatchSampler(
        args.env_name,
        batch_size=args.fast_batch_size,
        num_workers=args.num_workers
    )

    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape))
    )

    task_rewards = {}
    for i, task in enumerate(task_distribution):
        writer = SummaryWriter(os.path.join(writer_path, str(task['task_id'])))

        if continuous_actions:
            policy = MAESNNormalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                int(args.latent_dim),
                int(np.prod(sampler.envs.action_space.shape)),
                (args.hidden_size,) * args.num_layers,
                len(task_distribution),
                default_step_size=args.fast_lr,
                evaluate_mode=True
            )
            policy.load_state_dict(checkpoint)
        else:
            raise NotImplementedError

        metalearner = MAESNMetaLearner(
            sampler, policy, baseline,
            gamma=args.gamma, fast_lr=args.fast_lr,
            tau=args.tau, device=args.device
        )

        for batch in range(args.num_batches):
            start = time.time()
            sampler.reset_task(task)
            episodes = sampler.sample(
                policy,
                gamma=args.gamma,
                device=args.device
            )
            logger.debug('Finished sampling episodes in {:.3f} seconds.'.format(time.time() - start))

            start = time.time()
            params = metalearner.adapt(episodes)
            policy.set_parameters(params)
            logger.debug('Finished adaptation step in {:.3f} seconds.'.format(time.time() - start))

            if batch >= 1:
                with torch.no_grad():
                    policy.latent_mus_step_size /= 2
                    policy.latent_sigmas_step_size /= 2
                logger.debug('Reduced step sizes by half.')

            if not task['task_id'] in task_rewards:
                task_rewards[task['task_id']] = []

            rewards = torch_utils.total_rewards([episodes.rewards])
            task_rewards[task['task_id']].append(rewards)

            # Tensorboard
            writer.add_scalar(
                'meta-test/total_rewards',
                rewards,
                batch
            )

            writer.add_scalar(
                'meta-test/latent_mus_step_size',
                policy.latent_mus_step_size.mean(),
                batch
            )
            writer.add_scalar(
                'meta-test/latent_sigmas_step_size',
                policy.latent_sigmas_step_size.mean(),
                batch
            )

            latent_distribution = policy.latent_distribution(episodes.task_id)
            writer.add_scalar(
                'meta-test/latent_mus',
                latent_distribution.loc.mean(),
                batch
            )
            writer.add_scalar(
                'meta-test/latent_sigmas',
                latent_distribution.scale.mean(),
                batch
            )

        out = 'policy-{0}.pth.tar'.format(task['task_id'])
        with open(os.path.join(save_folder, out), 'wb') as f:
            torch.save(policy.state_dict(), f)

        logger.debug('Saved policy for task {0} to {1}.'.format(
            task['task_id'], os.path.join(save_folder, out)
        ))

    # Tensorboard
    writer = SummaryWriter(os.path.join(writer_path, 'aggregate'))

    average_rewards = [0.] * args.num_batches
    for i in range(args.num_batches):
        for task in task_distribution:
            average_rewards[i] += task_rewards[task['task_id']][i] # something is wrong here.

    average_rewards = [r / len(task_distribution) for r in average_rewards]
    for i, rewards in enumerate(average_rewards):
        writer.add_scalar(
            'meta-test/total_rewards',
            rewards,
            i
        )


if __name__ == '__main__':
    """

    """
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description='Meta-Reinforcement Learning with Structured Exploration (MAESN)'
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