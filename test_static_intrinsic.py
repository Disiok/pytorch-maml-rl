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
from maml_rl.policies import IntrinsicReward
from maml_rl.policies import NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.static_intrinsic_sampler import StaticIntrinsicBatchSampler
from maml_rl.static_intrinsic_metalearner import StaticIntrinsicMetaLearner
from maml_rl.utils import task_utils, torch_utils

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

    sampler = StaticIntrinsicBatchSampler(
        args.env_name,
        batch_size=args.fast_batch_size,
        num_workers=args.num_workers
    )

    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape))
    )

    task_extrinsic_rewards = {}
    task_intrinsic_rewards = {}

    for i, task in enumerate(task_distribution):
        writer = SummaryWriter(os.path.join(writer_path, str(task['task_id'])))

        # Construct the policy.
        if continuous_actions:
            policy = NormalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                int(np.prod(sampler.envs.action_space.shape)),
                (args.hidden_size,) * args.num_layers
            )

            if args.policy_checkpoint is not None:
                assert(os.path.exists(args.policy_checkpoint))
                policy_checkpoint = torch.load(args.policy_checkpoint)
                policy.load_state_dict(policy_checkpoint)
        else:
            raise NotImplementedError

        # Construct the reward network.
        reward = IntrinsicReward(
            int(np.prod(sampler.envs.observation_space.shape)) +
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,)*args.num_layers,
            reward_importance=args.intrinsic_weight
        )

        if args.reward_checkpoint is not None:
            assert(os.path.exists(args.reward_checkpoint))
            reward_checkpoint = torch.load(args.reward_checkpoint)
            reward.load_state_dict(reward_checkpoint)

        # Construct the metalearner.
        metalearner = StaticIntrinsicMetaLearner(
            sampler, policy, reward, baseline,
            gamma=args.gamma, fast_lr=args.fast_lr,
            tau=args.tau, device=args.device
        )

        train_episodes = None
        for batch in range(args.num_batches):
            sampler.reset_task(task)

            start = time.time()
            if train_episodes is None:
                train_episodes = sampler.sample(
                    policy,
                    gamma=args.gamma,
                    device=args.device
                )
            logger.debug('Finished sampling train episodes in {:.3f} seconds.'.format(time.time() - start))

            start = time.time()
            policy_params = metalearner.adapt_policy(train_episodes)
            logger.debug('Finished policy adaptation step in {:.3f} seconds.'.format(time.time() - start))

            start = time.time()
            val_episodes = sampler.sample(
                policy,
                policy_params,
                gamma=args.gamma,
                device=args.device
            )
            logger.debug('Finished sampling val episodes in {:.3f} seconds.'.format(time.time() - start))

            start = time.time()
            reward_params = metalearner.adapt_reward(val_episodes, policy_params=policy_params)
            logger.debug('Finished reward adaptation step in {:.3f} seconds.'.format(time.time() - start))

            policy.set_parameters(policy_params)
            reward.set_parameters(reward_params)

            # Compute extrinsic and intrinsic rewards.
            if not task['task_id'] in task_extrinsic_rewards:
                task_extrinsic_rewards[task['task_id']] = []
                task_intrinsic_rewards[task['task_id']] = []

            extrinsic_rewards = torch_utils.total_rewards([train_episodes.rewards])
            task_extrinsic_rewards[task['task_id']].append(extrinsic_rewards)

            intrinsic_rewards = torch_utils.total_rewards([train_episodes.intrinsic_rewards(reward)])
            task_intrinsic_rewards[task['task_id']].append(intrinsic_rewards)

            # Tensorboard
            writer.add_scalar(
                'meta-test/extrinsic_rewards',
                extrinsic_rewards,
                batch
            )
            writer.add_scalar(
                'meta-test/intrinsic_rewards',
                intrinsic_rewards,
                batch
            )
            writer.add_scalar(
                'meta-test/mixed_rewards',
                extrinsic_rewards + intrinsic_rewards,
                batch
            )

            for name, param in reward.named_parameters():
                writer.add_histogram('reward/' + name, param.detach().cpu().numpy(), batch)

            for name, param in policy.named_parameters():
                writer.add_histogram('policy/' + name, param.detach().cpu().numpy(), batch)

            # Set train_episodes as val_episodes
            train_episodes = val_episodes

        out = 'policy-{0}.pth.tar'.format(task['task_id'])
        with open(os.path.join(save_folder, out), 'wb') as f:
            torch.save(policy.state_dict(), f)

        logger.debug('Saved policy for task {0} to {1}.'.format(
            task['task_id'], os.path.join(save_folder, out)
        ))

        out = 'reward-{0}.pth.tar'.format(task['task_id'])
        with open(os.path.join(save_folder, out), 'wb') as f:
            torch.save(reward.state_dict(), f)

        logger.debug('Saved reward for task {0} to {1}.'.format(
            task['task_id'], os.path.join(save_folder, out)
        ))

    # Tensorboard
    writer = SummaryWriter(os.path.join(writer_path, 'aggregate'))

    average_extrinsic_rewards = [0.] * args.num_batches
    average_intrinsic_rewards = [0.] * args.num_batches
    average_mixed_rewards = [0.] * args.num_batches

    for i in range(args.num_batches):
        for task in task_distribution:
            average_extrinsic_rewards[i] += task_extrinsic_rewards[task['task_id']][i]
            average_intrinsic_rewards[i] += task_intrinsic_rewards[task['task_id']][i]
            average_mixed_rewards[i] += task_extrinsic_rewards[task['task_id']][i]
            average_mixed_rewards[i] += task_intrinsic_rewards[task['task_id']][i]

    average_extrinsic_rewards = [r / len(task_distribution) for r in average_extrinsic_rewards]
    average_intrinsic_rewards = [r / len(task_distribution) for r in average_intrinsic_rewards]
    average_mixed_rewards = [r / len(task_distribution) for r in average_mixed_rewards]

    for i in range(len(average_extrinsic_rewards)):
        writer.add_scalar(
            'meta-test/extrinsic_rewards',
            average_extrinsic_rewards[i],
            i
        )
        writer.add_scalar(
            'meta-test/intrinsic_rewards',
            average_intrinsic_rewards[i],
            i
        )
        writer.add_scalar(
            'meta-test/mixed_rewards',
            average_mixed_rewards[i],
            i
        )


if __name__ == '__main__':
    """

    """
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

    # Policy network (relu activation function)
    parser.add_argument('--policy-checkpoint', type=str, default=None,
        help='model checkpoint')
    parser.add_argument('--reward-checkpoint', type=str, default=None,
        help='reward checkpoint')
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')
    parser.add_argument('--intrinsic-weight', type=float, default=1.0,
        help='weight of intrinsic rewards')

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