#
#
#
import maml_rl.envs
import gym
import numpy as np
import torch
import json
import random

from maml_rl.maesn_metalearner import MAESNMetaLearner
from maml_rl.policies import MAESNNormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.maesn_sampler import MAESNBatchSampler

from tensorboardX import SummaryWriter


CONTINUOUS_ENVS = [
    'Pusher-v0',
    'AntVel-v1',
    'AntDir-v1',
    'AntPos-v0',
    'AntGoalRing-v0',
    'HalfCheetahVel-v1',
    'HalfCheetahDir-v1',
    '2DNavigation-v0',
]


def total_rewards(episodes_rewards, aggregation=torch.mean):
    """

    """
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()


def main(args):
    """

    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    continuous_actions = (args.env_name in CONTINUOUS_ENVS)
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    assert(os.path.exists(args.tasks))
    task_distribution = torch.load(args.tasks)

    sampler = MAESNBatchSampler(
        args.env_name,
        batch_size=args.fast_batch_size,
        num_workers=args.num_workers
    )

    if continuous_actions:
        policy = MAESNNormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(args.latent_dim),
            int(np.prod(sampler.envs.action_space.shape)),
            (args.hidden_size,) * args.num_layers,
            len(task_distribution),
            default_step_size=args.fast_lr
        )
    else:
        raise NotImplementedError

    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape))
    )

    metalearner = MAESNMetaLearner(
        sampler, policy, baseline,
        gamma=args.gamma, fast_lr=args.fast_lr,
        tau=args.tau, device=args.device
    )

    for batch in range(args.num_batches):
        tasks = random.sample(task_distribution, args.meta_batch_size)
        episodes = metalearner.sample(tasks, first_order=args.first_order)

        metalearner.step(
            episodes,
            max_kl=args.max_kl,
            cg_iters=args.cg_iters,
            cg_damping=args.cg_damping,
            ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio
        )

        # Tensorboard
        writer.add_scalar('total_rewards/before_update',
            total_rewards([ep.rewards for ep, _ in episodes]), batch)
        writer.add_scalar('total_rewards/after_update',
            total_rewards([ep.rewards for _, ep in episodes]), batch)

        writer.add_scalar('latent_space/latent_mus_step_size',
            policy.latent_mus_step_size.mean(), batch)
        writer.add_scalar('latent_space/latent_sigmas_step_size',
            policy.latent_sigmas_step_size.mean(), batch)

        writer.add_scalar('latent_space/latent_mus',
            policy.latent_mus.mean(), batch)
        writer.add_scalar('latent_space/latent_sigmas',
            policy.latent_sigmas.mean(), batch)

        # Save policy network
        save_file = os.path.join(save_folder, 'policy-{0}.pt'.format(batch))
        with open(save_file, 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
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
    parser.add_argument('--latent-dim', type=int, default=2,
        help='dimension of the latent space')

    # Task-specific
    parser.add_argument('--tasks', type=str, default=None,
        help='task distribution')
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