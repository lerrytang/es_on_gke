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
import cma
import gin
import numpy as np


@gin.configurable
class CMASolver(algorithm.solver.Solver):
    """Wrapper around pycma.

    Covariance Matrix Adaptation (CMA)
    Paper: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf
    Code: Based on https://github.com/hardmaru/estool/blob/master/es.py
    """

    def __init__(
            self, seed, policy, init_sigma, population_size, l2_coefficient):
        """Initialization.

        Instantiate the wrapper.

        Args:
            seed: int. Initial random seed.
            policy: An instance of ars.policy. The policy to optimize.
            init_sigma: Float. Initial standard deviation value.
            population_size: Integer. Size of the population.
            l2_coefficient: Float. Coefficient for the L2 regularization,
                a negative value disables L2 regularization.
        """
        self._policy = policy
        self._init_sigma = init_sigma
        self._population_size = population_size
        self._l2_coefficient = l2_coefficient
        self._solutions = None
        self._cma = cma.CMAEvolutionStrategy(
            self._policy.get_params(),
            self._init_sigma,
            inopts={'popsize': self._population_size,
                    'seed': seed if seed > 0 else 42,  # cma ignores if seed==0
                    'randn': np.random.randn})
        self._set_seed(seed)

    def ask(self, n_repeats=1):
        """Ask for parameters for a new generation.

        Return candidate parameters for the next generation.

        Args:
            n_repeats: int. For each parameter setting, test n_repeats times.
        Returns:
            np.array. Array of parameters for the next generation.
            list / np.array. Array of random seeds.
        """
        self._solutions = np.array(self._cma.ask())
        rand_seeds = self.gen_rand_int(self._solutions.shape[0] * n_repeats)
        return self._solutions, rand_seeds

    def tell(self, reported_rewards_table):
        """Report rewards from rollouts to CMA backend.

        This function gives the workers a chance to report collected utilities.

        Args:
            reported_rewards_table: list. A list of utilities collected
                in rollouts.
        """
        reward_table = -np.array(reported_rewards_table)
        if self._l2_coefficient > 0:
            l2_penalty = algorithm.solver.compute_l2_penalty(
                self._solutions, self._l2_coefficient)
            reward_table += l2_penalty
        self._cma.tell(self._solutions, reward_table.tolist())

    def get_current_params(self):
        """Return current parameters."""
        return self._cma.result.xfavorite

    def get_best_params(self):
        """Return parameters with best evaluation scores."""
        return self._cma.result.xbest
