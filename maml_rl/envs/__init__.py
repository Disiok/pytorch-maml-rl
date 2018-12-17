from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='maml_rl.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='maml_rl.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'AntVel-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntVelEnv'},
    max_episode_steps=200
)

register(
    'AntDir-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntDirEnv'},
    max_episode_steps=200
)

register(
    'AntPos-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntPosEnv'},
    max_episode_steps=200
)

register(
    'AntGoalRing-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntGoalRingEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahVelEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahDirEnv'},
    max_episode_steps=200
)

register(
    'Pusher-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.pusher:PusherTaskEnv'},
    max_episode_steps=200
)

register(
    'Wheeled-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.wheeled:WheeledTaskEnv'},
    max_episode_steps=200
)

register(
    'SparsePusher-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.pusher:PusherTaskEnv', 'sparse': True},
    max_episode_steps=200
)

register(
    'SparseAntGoalRing-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntGoalRingEnv', 'sparse': True},
    max_episode_steps=200
)

register(
    'SparseWheeled-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.wheeled:WheeledTaskEnv', 'sparse': True},
    max_episode_steps=200
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

CONTINUOUS_ENVS = [
    # MAML environments
    '2DNavigation-v0',
    'AntVel-v1',
    'AntDir-v1',
    'AntPos-v0',
    'HalfCheetahVel-v1',
    'HalfCheetahDir-v1',

    # MAESN dense environments
    'AntGoalRing-v0',
    'Pusher-v0',
    'Wheeled-v0',

    # MAESN sparse environments
    'SparseAntGoalRing-v0',
    'SparseWheeled-v0',
    'SparsePusher-v0',
]