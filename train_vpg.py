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
from maml_rl.policies import NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.utils import torch_utils, task_utils

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
    writer = SummaryWriter(os.path.join(args.out, 'logs'))
    save_folder = os.path.join(args.out, 'saves')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    assert(os.path.exists(args.tasks))
    task_distribution = task_utils.normalize_task_ids(torch.load(args.tasks))

    sampler = BatchSampler(
        args.env_name,
        batch_size=args.fast_batch_size,
        num_workers=args.num_workers
    )

    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers
        )

        if args.checkpoint is not None:
            assert(os.path.exists(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            policy.load_state_dict(checkpoint)
    else:
        raise NotImplementedError

    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape))
    )

    metalearner = MetaLearner(
        sampler, policy, baseline,
        gamma=args.gamma, fast_lr=args.fast_lr,
        tau=args.tau, device=args.device
    )

    for batch in range(args.num_batches):
        start = time.time()
        tasks = random.sample(task_distribution, args.meta_batch_size)
        logger.debug('Finished sampling tasks in {:.3f} seconds.'.format(time.time() - start))

        start = time.time()
        episodes = []
        for task in tasks:
            sampler.reset_task(task)
            episodes.append(sampler.sample(
                policy,
                gamma=args.gamma,
                device=args.device
            ))
        logger.debug('Finished sampling episodes in {:.3f} seconds.'.format(time.time() - start))

        start = time.time()
        total_loss = 0.0
        for i, task in enumerate(tasks):
            # Fit the baseline to the training episodes.
            baseline.fit(episodes[i])

            # Get the loss on the training episodes.
            task_loss = metalearner.inner_loss(episodes[i])
            total_loss += task_loss

        total_loss /= len(tasks)
        logger.debug('Finished computing losses in {:.3f} seconds.'.format(time.time() - start))

        start = time.time()
        params = policy.update_params(
            total_loss,
            step_size=args.fast_lr,
            first_order=True
        )
        policy.set_parameters(params)
        logger.debug('Finished updating policy in {:.3f} seconds.'.format(time.time() - start))

        # Tensorboard
        writer.add_scalar('total_rewards/before_update',
            torch_utils.total_rewards([ep.rewards for ep in episodes]), batch)

        for name, param in policy.named_parameters():
            writer.add_histogram('policy/' + name, param.detach().cpu().numpy(), batch)

        # Save policy network
        save_file = os.path.join(save_folder, 'policy-{0}.pt'.format(batch))
        with open(save_file, 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description='Meta-Pretraining for VPG.'
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
        help='checkpoint to start from')
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--tasks', type=str, default=None,
        help='task distribution')
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=500,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')

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