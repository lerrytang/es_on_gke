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
import numpy as np
import os


class Policy(object):
    """Base class for all policies."""

    def forward(self, inputs):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def set_params(self, params):
        raise NotImplementedError()

    def load_model(self, model_file):
        """Load model weights from a file."""
        data = np.load(model_file)
        params = data['params']
        self.set_params(params)

    def save_model(self, model_dir, checkpoint, best_model=False):
        """Save model to a directory."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_file = os.path.join(model_dir, 'model_{}.npz'.format(checkpoint))
        np.savez(model_file, params=self.get_params())
        if best_model:
            model_file = os.path.join(model_dir, 'best_model.npz')
            np.savez(model_file, params=self.get_params())


@gin.configurable
class MLP(Policy):
    """MLP model."""

    def __init__(self,
                 input_size,
                 output_size,
                 action_low,
                 action_high,
                 layers,
                 activation):
        """Initialization."""
        self.action_high = action_high
        self.action_low = action_low
        self.W = []
        self.b = []
        self.activations = []
        self.num_params = 0
        for layer_size in layers:
            self.W.append(np.zeros([layer_size, input_size]))
            self.b.append(np.zeros(layer_size))
            self.num_params += layer_size * (input_size + 1)
            input_size = layer_size
            if activation == 'relu':
                self.activations.append(lambda x: np.maximum(0, x))
            elif activation == 'tanh':
                self.activations.append(np.tanh)
            else:
                self.activations.append(lambda x: x)
        self.W.append(np.zeros([output_size, input_size]))
        self.b.append(np.zeros(output_size))
        self.num_params += output_size * (input_size + 1)
        self.activations.append(np.tanh)

    def forward(self, inputs):
        """Forward."""
        x = inputs
        for w, b, activation in zip(self.W, self.b, self.activations):
            x = activation(w.dot(x) + b)
        scaled_output = (x * (self.action_high - self.action_low) / 2.0 +
                         (self.action_high + self.action_low) / 2.0)
        return scaled_output

    def get_params(self):
        """Get all parameters."""
        params = [np.concatenate([x.copy().ravel(), y.copy()])
                  for x, y in zip(self.W, self.b)]
        return np.concatenate(params)

    def set_params(self, params):
        """Set parameters."""
        # round parameters in the hope that the policy will not be sensitive
        params = np.round(params, decimals=4)

        n = len(self.W)
        offset = 0
        for i in range(n):
            o_size, i_size = self.W[i].shape
            param_size = o_size * i_size
            self.W[i] = params[offset:(offset + param_size)].reshape(
                [o_size, i_size])
            offset += param_size
            self.b[i] = params[offset:(offset + o_size)]
            offset += o_size
