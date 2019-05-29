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


import argparse
import gin
from gym import wrappers
import os
import utility
import pandas as pd


def main(config):
    """Test policy."""

    rewards = []
    steps = []

    env = utility.load_env()
    env = wrappers.Monitor(
        env, config.video_dir, force=True, video_callable=lambda x: x < 3)
    env.seed(config.seed)

    policy = utility.create_policy(env)
    model_file = os.path.join(config.log_dir, 'model_{}.npz'.format(
        config.checkpoint))
    policy.load_model(model_file)

    for i in range(config.n_episodes):
        ob = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        while not done:
            action = policy.forward(ob)
            ob, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if config.render:
                env.render()
        print('reward={0:.2f}, steps={1}'.format(ep_reward, ep_steps))
        rewards.append(ep_reward)
        steps.append(ep_steps)

    result_df = pd.DataFrame({'reward': rewards, 'step': steps})
    result_df['avg_reward_per_step'] = result_df.reward / result_df.step
    result_df.to_csv(
        os.path.join(config.log_dir, 'test_scores.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-dir', help='Directory of logs.')
    parser.add_argument(
        '--checkpoint', help='Checkpoint of the model to evaluate.')
    parser.add_argument(
        '--video-dir', help='Directory to save videos.', default='./video')
    parser.add_argument(
        '--render', help='Whether to render while evaluation.', default=False,
        action='store_true')
    parser.add_argument(
        '--n-episodes', help='Number of episodes to evaluate.',
        type=int, default=3)
    parser.add_argument(
        '--seed', help='Random seed for evaluation.', type=int, default=42)
    args, _ = parser.parse_known_args()

    gin.parse_config_file(os.path.join(args.log_dir, 'config.gin'))
    try:
        gin.query_parameter('utility.load_env.render')
        gin.bind_parameter("utility.load_env.render", args.render)
    except ValueError as e:
        gin.bind_parameter("utility.load_env.renders", args.render)

    main(args)
