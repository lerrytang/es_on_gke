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
import evaluation_service_pb2
import evaluation_service_pb2_grpc
import numpy as np
import threading
import utility


@gin.configurable
class RolloutServicer(evaluation_service_pb2_grpc.RolloutServicer):
    """Object class for parallel rollout generation."""

    def __init__(self, worker_id, max_steps):
        """Initialization."""

        self.lock = threading.Lock()
        self.env = utility.load_env()
        self.policy = utility.create_policy(self.env)
        self.worker_id = worker_id
        self.max_steps = max_steps
        self.logger = utility.create_logger()
        self.logger.info('Rollout service initialized (id={}).'.format(
            self.worker_id))

    def RolloutWithParameter(self, request, context):
        """Perform rollouts with the given parameters."""

        # Whenever a request comes, gRPC spawns another thread to handle it.
        # The thread can set env.seed and that affects the trials run by others,
        # the result of which is un-reproducible sequence of scores.
        # Thus, set a lock here.
        # TODO: find a better way to handle this.
        try:
            self.lock.acquire(blocking=True)
            policy_weights = np.asarray(request.policy_weights)
            self.policy.set_params(policy_weights)
            self.env.seed(request.env_seed)
            ep_reward = self._Rollout(
                env=self.env, policy=self.policy, evaluate=request.evaluate)
        finally:
            self.lock.release()
        return evaluation_service_pb2.RolloutResponse(
            rollout_index=request.rollout_index,
            rollout_reward=ep_reward)

    def _Rollout(self, env, policy, evaluate):
        """Performs one rollout.

        Args:
          env: the env to perform rollouts.
          policy: policy to use for the rollouts.
          evaluate: bool. Whether this rollout is an evaluation,
        Returns:
          the total reward of this rollout.
        """

        ep_reward = 0.
        ob = env.reset()
        for _ in range(self.max_steps):
            action = policy.forward(ob)
            ob, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                if not evaluate and reward == -100:
                    # Hack for BipedalWalkerHardcore-v2
                    ep_reward -= reward
                break
        return ep_reward
