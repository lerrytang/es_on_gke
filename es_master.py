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
import evaluation_service_pb2_grpc
import gin
import grpc
import learner
import time


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def main(config):
    """Train an ES policy."""
    if config.run_on_gke:
        servers = ['{}'.format(addr)
                   for addr in config.server_addresses.split(',')]
    else:
        servers = ['127.0.0.1:{}'.format(config.server_port + i)
                   for i in range(config.num_workers)]
    print(servers)
    stubs = []
    for server in servers:
        if config.run_on_gke:
            channel = grpc.insecure_channel(
                server, [('grpc.lb_policy_name', 'round_robin')])
        else:
            channel = grpc.insecure_channel(server)
        grpc.channel_ready_future(channel).result()
        stubs.append(evaluation_service_pb2_grpc.RolloutStub(channel))

    learner.ESLearner(
        logdir=config.logdir,
        config=config.config,
        stubs=stubs,
    ).train()

    # This is to prevent GKE from restarting pod when the job finishes.
    if config.run_on_gke:
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            print('Job done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', help='Path to the config file.')
    parser.add_argument(
        '--logdir', help='Path to the log directory.', default='./log')
    parser.add_argument(
        '--num-workers', help='Number of workers.', type=int, default=48)
    parser.add_argument(
        '--server-port', help='Port of servers.', type=int, default=20000)
    parser.add_argument(
        '--server-addresses', help='Server addresses, separated by comma.')
    parser.add_argument(
        '--run-on-gke', help='Whether run this on GKE.', default=False,
        action='store_true')
    args, _ = parser.parse_known_args()

    gin.parse_config_file(args.config)
    main(args)
