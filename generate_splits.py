#
#
#

import os
import torch
import random
import argparse
import numpy as np

from maml_rl.maesn_sampler import MAESNBatchSampler


CONTINUOUS_ENVS = [
    'Wheeled-v0',
    'Pusher-v0',
    'AntVel-v1',
    'AntDir-v1',
    'AntPos-v0',
    'AntGoalRing-v0',
    'HalfCheetahVel-v1',
    'HalfCheetahDir-v1',
    '2DNavigation-v0',
]


def main(args):
    """

    """
    assert(args.env_name in CONTINUOUS_ENVS)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sampler = MAESNBatchSampler(
        args.env_name,
        batch_size=100,
        num_workers=8,
    )

    num_tasks = args.num_train_tasks + args.num_val_tasks
    task_distribution = sampler.sample_tasks(num_tasks=num_tasks)

    train_tasks = task_distribution[:args.num_train_tasks]
    val_tasks = task_distribution[args.num_train_tasks:]

    train_out = '{0}_train.pth.tar'.format(args.env_name)
    torch.save(train_tasks, os.path.join(args.out, train_out))

    val_out = '{0}_val.pth.tar'.format(args.env_name)
    torch.save(val_tasks, os.path.join(args.out, val_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate splits.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env-name', type=str, default=None)
    parser.add_argument('--num-train-tasks', type=int, default=100)
    parser.add_argument('--num-val-tasks', type=int, default=100)
    parser.add_argument('--out', type=str, default='debug')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    main(args)