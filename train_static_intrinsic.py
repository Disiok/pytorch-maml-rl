#
#
#
import gym
import json
import time
import torch
import random
import logging
import numpy as np

import maml_rl.envs
from maml_rl.envs import CONTINUOUS_ENVS
from maml_rl.policies import NormalMLPPolicy, IntrinsicReward
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.static_intrinsic_sampler import StaticIntrinsicBatchSampler
from maml_rl.static_intrinsic_metalearner import StaticIntrinsicMetaLearner
from maml_rl.utils import torch_utils

from tensorboardX import SummaryWriter
from torch.nn.utils.convert_parameters import (
    vector_to_parameters, parameters_to_vector
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def normalize_task_ids(task_distribution):
    """
    Normalize task ids.

    :param task_distribution [list<dict>]: A list of task configurations.
    :return                  [list<dict>]: A list of task configurations.
    """
    task_distribution = sorted(task_distribution, key=lambda t: t['task_id'])
    for task_id, task in enumerate(task_distribution):
        task['task_id'] = task_id
    return task_distribution


def main(args):
    """

    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert(args.env_name in CONTINUOUS_ENVS)
    writer = SummaryWriter(os.path.join(args.out, 'logs'))
    save_folder = os.path.join(args.out, 'saves')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    assert(os.path.exists(args.tasks))
    task_distribution = normalize_task_ids(torch.load(args.tasks))

    sampler = StaticIntrinsicBatchSampler(
        args.env_name,
        batch_size=args.fast_batch_size,
        num_workers=args.num_workers
    )

    policy = NormalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        int(np.prod(sampler.envs.action_space.shape)),
        (args.hidden_size,) * args.num_layers
    )
    reward = IntrinsicReward(
        int(np.prod(sampler.envs.observation_space.shape)) +
        int(np.prod(sampler.envs.action_space.shape)),
        hidden_sizes=(args.hidden_size,)*args.num_layers,
        reward_importance=args.intrinsic_weight
    )

    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape))
    )

    metalearner = StaticIntrinsicMetaLearner(
        sampler, policy, reward, baseline,
        gamma=args.gamma, fast_lr=args.fast_lr,
        tau=args.tau, device=args.device
    )

    for batch in range(args.num_batches):
        start = time.time()
        tasks = random.sample(task_distribution, args.meta_batch_size)
        logger.debug('Finished sampling tasks in {:.3f} seconds.'.format(time.time() - start))

        start = time.time()
        episodes = metalearner.sample(tasks, first_order=args.first_order)
        logger.debug('Finished sampling episodes in {:.3f} seconds.'.format(time.time() - start))

        log_episodes(writer, episodes, reward, batch)

        start = time.time()
        policy_step, reward_step = metalearner.step(
            episodes,
            max_kl=args.max_kl,
            cg_iters=args.cg_iters,
            cg_damping=args.cg_damping,
            ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio
        )
        logger.debug('Finished metalearner step in {:.3f} seconds.'.format(time.time() - start))

        log_policies(writer, policy, reward, policy_step, reward_step, batch)
        save_policies(policy, reward, save_folder)


def save_policies(policy, reward, save_folder):
    # Save policy network
    save_file = os.path.join(save_folder, 'policy-{0}.pt'.format(batch))
    with open(save_file, 'wb') as f:
        torch.save(policy.state_dict(), f)

    # Save reward network
    save_file = os.path.join(save_folder, 'reward-{0}.pt'.format(batch))
    with open(save_file, 'wb') as f:
        torch.save(reward.state_dict(), f)


def log_policies(writer, policy, reward, policy_step, reward_step, batch):
    for name, param in reward.named_parameters():
        writer.add_histogram('reward/' + name, param.detach().cpu().numpy(), batch)

    for name, param in policy.named_parameters():
        writer.add_histogram('policy/' + name, param.detach().cpu().numpy(), batch)

    policy_parameters = []
    for (name, param) in policy.named_parameters():
        policy_parameters.append(param.clone())

    reward_parameters = []
    for (name, param) in reward.named_parameters():
        reward_parameters.append(param.clone())

    vector_to_parameters(policy_step.detach(), policy_parameters)
    vector_to_parameters(reward_step.detach(), reward_parameters)

    for (name, param), grad in zip(reward.named_parameters(), reward_parameters):
        writer.add_histogram('reward_grad/' + name, grad.detach().cpu().numpy(), batch)

    for (name, param), grad in zip(policy.named_parameters(), policy_parameters):
        writer.add_histogram('policy_grad/' + name, grad.detach().cpu().numpy(), batch)


def log_episodes(writer, episodes, reward, batch):
    writer.add_scalar(
        'intrinsic_rewards/before_update',
        torch_utils.total_rewards([ep.intrinsic_rewards(reward) for ep, _ in episodes]),
        batch
    )
    writer.add_scalar(
        'intrinsic_rewards/after_update',
        torch_utils.total_rewards([ep.intrinsic_rewards(reward) for _, ep in episodes]),
        batch
    )

    writer.add_scalar(
        'mixed_rewards/before_update',
        torch_utils.total_rewards([ep.intrinsic_rewards(reward) + ep.rewards for ep, _ in episodes]),
        batch
    )
    writer.add_scalar(
        'mixed_rewards/after_update',
        torch_utils.total_rewards([ep.intrinsic_rewards(reward) + ep.rewards for _, ep in episodes]),
        batch
    )

    writer.add_scalar(
        'sparse_rewards/before_update',
        torch_utils.total_rewards([ep.sparse_rewards for ep, _ in episodes]),
        batch
    )
    writer.add_scalar(
        'sparse_rewards/after_update',
        torch_utils.total_rewards([ep.sparse_rewards for _, ep in episodes]),
        batch
    )

    writer.add_scalar('total_rewards/before_update',
        torch_utils.total_rewards([ep.rewards for ep, _ in episodes]), batch)

    writer.add_scalar('total_rewards/after_update',
        torch_utils.total_rewards([ep.rewards for _, ep in episodes]), batch)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description='Meta-Reinforcement Learning with Intrinsic Rewards'
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
    parser.add_argument('--intrinsic-weight', type=float, default=1.0,
        help='weight of intrinsic rewards')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
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
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

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
