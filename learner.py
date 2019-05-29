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


import evaluation_service_pb2
import gin
import grpc
import numpy as np
import utility
import time


_TIMEOUT = 3600  # seconds


@gin.configurable
class ESLearner(object):
    """ES learner."""

    def __init__(self,
                 max_iters,
                 eval_n_episodes,
                 eval_every_n_iters,
                 population_size,
                 target_reward,
                 n_repeats,
                 solver,
                 logdir,
                 config,
                 stubs,
                 warm_start_from=None):
        """Initialization."""

        utility.save_config(logdir, config)
        self.logger = utility.create_logger(logdir)
        self.logdir = logdir
        self.env = utility.load_env()
        self.workers = stubs
        self.num_workers = len(self.workers)
        self.max_iters = max_iters
        self.population_size = population_size
        self.eval_n_episodes = eval_n_episodes
        self.eval_every_n_iters = eval_every_n_iters
        self.target_reward = target_reward
        self.n_repeats = n_repeats
        self.policy = utility.create_policy(self.env)
        if warm_start_from is not None:
            try:
                self.policy.load_model(warm_start_from)
                self.logger.info(
                    'Model loaded, continue training from {}.'.format(
                        warm_start_from))
            except IOError as e:
                self.logger.error('Failed to load model from {}: {}'.format(
                    warm_start_from, e))
        self.w_policy = self.policy.get_params()
        self.solver = solver(policy=self.policy,
                             population_size=self.population_size)
        self.best_score = -float('Inf')
        self.logger.info("Initialization of ESLearner complete")

    def evaluate_policies(self, rand_seeds, w_policy, evaluate=False):
        """Evaluate policies."""

        if evaluate:
            num_rollouts = self.eval_n_episodes
        else:
            num_rollouts = self.population_size * self.n_repeats
        self.logger.info('num_rollouts={}, len(rand_seeds)={}'.format(
            num_rollouts, len(rand_seeds)))

        unfinished_tasks = list(range(num_rollouts))
        rollout_rewards = np.zeros(num_rollouts)
        next_stub_id = 0

        while unfinished_tasks:

            # send evaluation requests
            futures = []
            for task_id in unfinished_tasks:
                if not evaluate:
                    policy_index = int(task_id / self.n_repeats)
                    policy_weight = w_policy[policy_index]
                else:
                    policy_weight = w_policy

                request = evaluation_service_pb2.RolloutRequest(
                    rollout_index=task_id,
                    env_seed=rand_seeds[task_id],
                    policy_weights=policy_weight.tolist(),
                    evaluate=evaluate)
                future = self.workers[next_stub_id].RolloutWithParameter.future(
                    request, timeout=_TIMEOUT)
                futures.append(future)
                next_stub_id = (next_stub_id + 1) % self.num_workers

            # collect results
            for future in futures:
                try:
                    result = future.result()
                    ix = result.rollout_index
                    rollout_rewards[ix] = result.rollout_reward
                    unfinished_tasks.remove(ix)
                except grpc.RpcError as e:
                    self.logger.error("RPC error caught in evaluate_policies!")
                    self.logger.error(e)

        if not evaluate:
            rollout_rewards = rollout_rewards.reshape(
                [-1, self.n_repeats]).mean(axis=-1)
        if not evaluate:
            assert rollout_rewards.size == self.population_size
        else:
            assert rollout_rewards.size == self.eval_n_episodes
        return rollout_rewards

    def _train_step(self):
        """Perform one update step of the policy weights."""
        self.logger.info('Ask for solution ...')
        start_time = time.time()
        params, rand_seeds = self.solver.ask(self.n_repeats)
        self.logger.info('Got solution, time={0:.2f}s'.format(
            time.time() - start_time))
        rewards = self.evaluate_policies(
            rand_seeds=rand_seeds,
            w_policy=params,
            evaluate=False)
        rewards = np.round(rewards, decimals=2)
        self.solver.tell(rewards)
        return rewards

    def _save_checkpoint_and_statistics(self, iter_count):
        """Save checkpoint for the policy and learning statistics.

        Args:
            iter_count: The number of iterations that have been finished.
        """
        rand_seeds = self.solver.gen_rand_int(self.eval_n_episodes)
        rewards = self.evaluate_policies(
            rand_seeds=rand_seeds,
            w_policy=self.policy.get_params(),
            evaluate=True)
        mean_score = np.mean(rewards)
        if mean_score > self.best_score:
            self.best_score = mean_score
            best_model = True
        else:
            best_model = False
        self.policy.save_model(self.logdir, iter_count, best_model)
        utility.log_rewards(self.logger, iter_count, rewards, True)
        utility.save_scores(self.logdir, iter_count, rewards)
        return rewards

    def train(self):
        """Train the agent using ES solver."""

        self._save_checkpoint_and_statistics(iter_count=0)

        for i in range(self.max_iters):
            rewards = self._train_step()
            utility.log_rewards(self.logger, i + 1, rewards)

            # record statistics every n iterations
            if (i + 1) % self.eval_every_n_iters == 0:
                params = self.solver.get_current_params()
                self.policy.set_params(params)
                rewards = self._save_checkpoint_and_statistics(
                    iter_count=i + 1)
                if np.mean(rewards) >= self.target_reward > 0:
                    self.logger.info('Target reward acquired! Stop training.')
                    break
