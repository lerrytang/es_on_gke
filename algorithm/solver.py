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


import numpy as np
import random


MAX_INT = (1 << 31) - 1


def compute_l2_penalty(solution, l2_coefficient):
    """Calculate L2 penalty based on policy parameters and coefficient.

    Calculate L2 penalty.

    Args:
        solution: list / np.array. Neural network parameters.
        l2_coefficient: float. Weight decay for L2 penalty.
    Returns:
        float. L2 penalty.
    """
    params = np.array(solution)
    return -l2_coefficient * np.mean(params * params, axis=1)


def compute_ranks(x):
    """Compute ranks.

    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """Compute centered ranks.

    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


class Solver(object):
    """Evolution strategy base solver."""

    def _set_seed(self, seed):
        """Set initial random seed.

        Set initial random seed, and fix seeds for all possible random number
        generators.

        Args:
            seed: int. Initial random seed.
        """
        np.random.seed(seed)
        random.seed(seed)
        self._seed = seed

    def gen_rand_int(self, size):
        """Get a list of random integers.

        Get a list of random integers.

        Args:
            size: int. Number of integers desired.
        Returns:
            list, a list of random integers.
        """
        return np.random.randint(MAX_INT, size=size).tolist()

    def ask(self, n_repeats=1):
        """Ask for parameters for a new generation.

        Return candidate parameters for the next generation.

        Args:
            n_repeats: int. For each parameter setting, test n_repeats times.
        Returns:
            np.array. Array of parameters for the next generation.
            list / np.array. Array of random seeds.
        """
        raise NotImplementedError()

    def tell(self, reported_rewards_table):
        """Report rewards from rollouts to CMA backend.

        This function gives the workers a chance to report collected utilities.

        Args:
            reported_rewards_table: list. A list of utilities collected
                in rollouts.
        """
        raise NotImplementedError()

    def get_current_params(self):
        """Return current parameters.

        Return current parameters.

        Returns:
            np.array. Current parameters.
        """
        raise NotImplementedError()

    def get_best_params(self):
        """Return parameters with best evaluation scores.

        Return parameters with best evaluation scores.

        Returns:
            np.array. Parameters with best evaluation scores.
        """
        raise NotImplementedError()
