# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import algorithm.solver
import algorithm.optimizers
import gin
import numpy as np


@gin.configurable
class PEPGSolver(algorithm.solver.Solver):
    """Parameter-exploring Policy Gradients (PEPG).

    Code: Based on https://github.com/hardmaru/estool/blob/master/es.py
    """

    def __init__(self,
                 seed,
                 policy,
                 sigma_init=0.10,
                 sigma_alpha=0.20,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 sigma_max_change=0.2,
                 learning_rate=0.01,
                 learning_rate_decay=0.9999,
                 learning_rate_limit=0.01,
                 elite_ratio=0,
                 population_size=256,
                 average_baseline=True,
                 weight_decay=0.01,
                 rank_fitness=True,
                 forget_best=True):
        """Initialization.

        Initialization.

        Args:
            seed: int. Initial random seed.
            policy: policies.Policy. Policy network.
            sigma_init: float. Initial standard deviation.
            sigma_alpha: float. Learning rate for standard deviation.
            sigma_decay: float. Anneal standard deviation.
            sigma_limit: float. Stop annealing if less than this.
            sigma_max_change: float. Clips adaptive sigma to this.
            learning_rate: float. Learning rate for standard deviation.
            learning_rate_decay: float. Anneal the learning rate.
            learning_rate_limit: float. Stop annealing learning rate.
            elite_ratio: float. If > 0, then ignore learning_rate.
            population_size: int. Population size.
            average_baseline: bool. Whether to set baseline to average of batch.
            weight_decay: float. Weight decay coefficient.
            rank_fitness: bool. Use rank rather than fitness numbers.
            forget_best: bool. Whether to forget historical best solution.
        """
        self._policy = policy
        self.num_params = self._policy.get_params().size
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.sigma_max_change = sigma_max_change
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = population_size
        self.average_baseline = average_baseline
        if self.average_baseline:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.batch_size = int(self.popsize / 2)
        else:
            assert (self.popsize & 1), "Population size must be odd"
            self.batch_size = int((self.popsize - 1) / 2)

        # option to use greedy es method to select next mu,
        # rather than using drift param
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.use_elite = False
        if self.elite_popsize > 0:
            self.use_elite = True

        self.forget_best = forget_best
        self.batch_reward = np.zeros(self.batch_size * 2)
        self.mu = self._policy.get_params()
        self.sigma = np.ones(self.num_params) * self.sigma_init
        self.curr_best_mu = self._policy.get_params()
        self.best_mu = self._policy.get_params()
        self.best_reward = 0
        self.first_interation = True
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = algorithm.optimizers.Adam(self, learning_rate)
        self._set_seed(seed)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self, n_repeats=1):
        """Ask for parameters for a new generation.

        Return candidate parameters for the next generation.

        Args:
            n_repeats: int. For each parameter setting, test n_repeats times.
        Returns:
            np.array. Array of parameters for the next generation.
            list / np.array. Array of random seeds.
        """
        # antithetic sampling
        self.epsilon = (np.random.randn(self.batch_size, self.num_params) *
                        self.sigma.reshape(1, self.num_params))
        self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
        if self.average_baseline:
            epsilon = self.epsilon_full
        else:
            # first population is mu, then positive epsilon,
            # then negative epsilon
            epsilon = np.concatenate(
                [np.zeros((1, self.num_params)), self.epsilon_full])
        solutions = self.mu.reshape(1, self.num_params) + epsilon
        self.solutions = solutions
        rand_seeds = self.gen_rand_int(self.batch_size * n_repeats)
        rand_seeds += rand_seeds
        return solutions, rand_seeds

    def tell(self, reward_table_result):
        """Report rewards from rollouts to CMA backend.

        This function gives the workers a chance to report collected utilities.

        Args:
            reward_table_result: list. A list of utilities collected
                in rollouts.
        """

        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), \
            "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.rank_fitness:
            reward_table = algorithm.solver.compute_centered_ranks(reward_table)

        if self.weight_decay > 0:
            l2_decay = algorithm.solver.compute_l2_penalty(
                self.solutions, self.weight_decay)
            reward_table += l2_decay

        reward_offset = 1
        if self.average_baseline:
            b = np.mean(reward_table)
            reward_offset = 0
        else:
            b = reward_table[0]  # baseline

        reward = reward_table[reward_offset:]
        if self.use_elite:
            idx = np.argsort(reward)[::-1][0:self.elite_popsize]
        else:
            idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        if (best_reward > b or self.average_baseline):
            best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            best_mu = self.mu
            best_reward = b

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.sigma = np.ones(self.num_params) * self.sigma_init
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # short hand
        epsilon = self.epsilon
        sigma = self.sigma

        # update the mean

        # move mean to the average of the best idx means
        if self.use_elite:
            self.mu += self.epsilon_full[idx].mean(axis=0)
        else:
            rT = (reward[:self.batch_size] - reward[self.batch_size:])
            change_mu = np.dot(rT, epsilon)
            self.optimizer.stepsize = self.learning_rate
            update_ratio = self.optimizer.update(
                -change_mu)  # adam, rmsprop, momentum, etc.
            # self.mu += (change_mu * self.learning_rate) # normal SGD method

        # adaptive sigma
        # normalization
        if self.sigma_alpha > 0:
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = reward.std()
            S = ((epsilon * epsilon - (sigma * sigma).reshape(
                1, self.num_params)) / sigma.reshape(1, self.num_params))
            reward_avg = (reward[:self.batch_size] + reward[
                                                     self.batch_size:]) / 2.0
            rS = reward_avg - b
            delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

            # adjust sigma according to the adaptive sigma calculation
            # for stability, don't let sigma move more than 10% of orig value
            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.minimum(change_sigma,
                                      self.sigma_max_change * self.sigma)
            change_sigma = np.maximum(change_sigma,
                                      - self.sigma_max_change * self.sigma)
            self.sigma += change_sigma

        if self.sigma_decay < 1:
            self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

        if (
                (self.learning_rate_decay < 1) and
                (self.learning_rate > self.learning_rate_limit)
        ):
            self.learning_rate *= self.learning_rate_decay

    def get_current_params(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def get_best_params(self):
        return self.best_mu

    def result(self):
        """Return parameters.

        Return parameter with best evaluation scores,
        along with historically best reward, curr reward, sigma
        """
        return (self.best_mu,
                self.best_reward,
                self.curr_best_reward,
                self.sigma)
