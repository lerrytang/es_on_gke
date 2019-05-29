# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gin
import gym
import logging
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pandas as pd
import pybullet_envs.bullet
import seaborn as sns
import shutil
import time


@gin.configurable
def load_env(env_package, env_name, **kwargs):
    """Load and return an environment.

    This function loads a gym environment.

    Args:
        env_package: str. Name of the package that contains the environment.
            If env_name is 'NULL', it is a built-in gym environment.
        env_name: str. Name of the environment.
        kwargs: dict. Environment configurations.
    Returns:
        gym.Environment.
    """

    if env_package == 'NULL':
        env = gym.make(env_name)
    elif env_package == 'CUSTOMIZED':
        # The customized env must be gin.configurable
        env = env_name(**kwargs)
    else:
        pkg = getattr(pybullet_envs.bullet, env_package)
        env = getattr(pkg, env_name)(**kwargs)
        if not hasattr(env, '_cam_dist'):
            env._cam_dist = 6
            env._cam_yaw = 0
            env._cam_pitch = -30

        # Some pybullet_env do not have close() implemented, add close()
        def close():
            if hasattr(env, '_pybullet_client'):
                env._pybullet_client.resetSimulation()
                del env._pybullet_client

        env.close = close

        # Some pybullet env do not have seed() implemented, add seed()
        def seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)

        env.seed = seed
    return env


@gin.configurable
def create_policy(env, policy_type, policy_weights_file=None):
    """Create a policy.

    Create and return a policy.

    Args:
        env: gym.Environment. Gym environment.
        policy_type: policies.X. Policy type.
        policy_weights_file: str. Path to policy weights file.
    Return:
        policies.policy.
    """
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    policy = policy_type(input_size=input_size,
                         output_size=output_size,
                         action_high=action_high,
                         action_low=action_low)
    if policy_weights_file:
        policy.load_model(policy_weights_file)
    return policy


def save_config(log_dir, config):
    """Create a log directory and save config in it.

    Create a log directory and save configurations.

    Args:
        log_dir: str. Path of the log directory.
        config: str. Path to configuration file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    shutil.copy(config, os.path.join(log_dir, 'config.gin'))


def create_logger(log_dir=None):
    """Create a logger.

    Create a logger that logs to logdir.

    Args:
        log_dir: str. Path to log directory.
    Returns:
        logging.logger.
    """
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(process)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger('es_on_gke')
    if log_dir:
        log_file = os.path.join(log_dir, 'log.txt')
        file_hdl = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt=log_format)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    return logger


def save_scores(log_dir, n_iter, scores):
    """Save scores to file system.

    Save scores to file system as a csv file.

    Args:
        log_dir: str. Path to log directory.
        n_iter: int. Current iteration number.
        scores: np.array. Scores.
    """
    # save scores for analysis in the future
    filename = os.path.join(log_dir, 'scores.csv')
    df = pd.DataFrame({'Time': [int(time.time())] * scores.size,
                       'Iteration': [n_iter] * scores.size,
                       'Reward': scores})
    need_header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', header=need_header, index=False)
    # draw graphs
    df = pd.read_csv(filename)
    sns.set()
    # plot 1: reward vs iteration
    sns_plot = sns.lineplot(x='Iteration', y='Reward', data=df, ci='sd')
    im_filename = os.path.join(log_dir, 'reward_vs_iteration.png')
    sns_plot.get_figure().savefig(im_filename)
    plt.clf()
    # plot 2: reward vs time
    start_time = df.Time.values[0]
    df.Time = (df.Time - start_time) / 3600  # convert to hours
    sns_plot = sns.lineplot(x='Time', y='Reward', data=df, ci='sd')
    sns_plot.set_xlabel('Time (hour)')
    im_filename = os.path.join(log_dir, 'reward_vs_time.png')
    sns_plot.get_figure().savefig(im_filename)
    plt.clf()


def log_rewards(logger, iter_cnt, rewards, evaluate=False):
    """Log rewards.

    Log rewards.

    Args:
        logger: A logger.
        iter_cnt: int. Iteration number.
        rewards: list. List of rewards.
        evaluate: bool. Whether these rewards are from evaluation rollouts.
    """
    msg = ('Iter {0}: max(reward)={1:.2f}, '
           'mean(reward)={2:.2f}, '
           'min(reward)={3:.2f}, '
           'sd(reward)={4:.2f}'.format(iter_cnt,
                                       np.max(rewards),
                                       np.mean(rewards),
                                       np.min(rewards),
                                       np.std(rewards)))
    if evaluate:
        msg = '[TEST] ' + msg
    logger.info(msg)
