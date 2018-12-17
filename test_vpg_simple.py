import maml_rl.envs
import gym
import numpy as np
import torch
import json
import logging
import time
from torch import optim
from IPython import embed

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy, IntrinsicReward
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.envs import CONTINUOUS_ENVS

from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def inner_loss(policy, baseline, episodes, params=None):
    """Compute the inner loss for the one-step gradient update. The inner 
    loss is REINFORCE with baseline [2], computed on advantages estimated 
    with Generalized Advantage Estimation (GAE, [3]).
    """
    values = baseline(episodes)
    advantages = episodes.gae(values, tau=1.0)
    advantages = weighted_normalize(advantages, weights=episodes.mask)

    pi = policy(episodes.observations, params=params)
    log_probs = pi.log_prob(episodes.actions)
    if log_probs.dim() > 2:
        log_probs = torch.sum(log_probs, dim=2)
    loss = -weighted_mean(log_probs * advantages, dim=0,
        weights=episodes.mask)
    
    if loss < -400:
        embed()

    return loss

def main(args):
    continuous_actions = (args.env_name in CONTINUOUS_ENVS)

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)

    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
        
        if args.intrinsic_reward:
            intrinsic_reward = IntrinsicReward(
                int(np.prod(sampler.envs.observation_space.shape)) + int(np.prod(sampler.envs.action_space.shape)),
                hidden_sizes=(args.hidden_size,) * args.num_layers)
        else:
            intrinsic_reward = None
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)

        if args.intrinsic_reward:
            intrinsic_reward = IntrinsicReward(
                int(np.prod(sampler.envs.observation_space.shape)) + sampler.envs.action_space.n,
                hidden_sizes=(args.hidden_size,) * args.num_layers)
        else:
            intrinsic_reward = None
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))
    optimizer = optim.SGD(policy.parameters(), lr=args.fast_lr)

    task = sampler.sample_tasks(num_tasks=5, seed=999)[0]
    for batch in range(args.num_batches):
        logger.debug('Processing batch {}'.format(batch))

        sampler.reset_task(task)
        train_episodes = sampler.sample(policy, gamma=args.gamma, device=args.device)

        # Fit the baseline to the training episodes
        baseline.fit(train_episodes)
        # Get the loss on the training episodes
        optimizer.zero_grad()
        loss = inner_loss(policy, baseline, train_episodes)
        loss.backward()
        optimizer.step()
        episodes = [train_episodes]

        # Tensorboard
        writer.add_scalar('meta-test/loss', loss.item(), batch)
        writer.add_scalar('meta-test/total_rewards',
            total_rewards([ep.rewards for ep in episodes]), batch)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')
    
    # Intrinsic reward network
    parser.add_argument('--intrinsic-reward', action='store_true',
        help='use intrinsic reward to provide additional supervision')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
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
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
