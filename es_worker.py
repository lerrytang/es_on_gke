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
from concurrent import futures
import gin
import evaluation_service
import evaluation_service_pb2_grpc
import time
import grpc


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def main(config):
    """Start evaluation service."""

    if config.run_on_gke:
        port = config.port
    else:
        port = config.port + config.worker_id
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = evaluation_service.RolloutServicer(
        worker_id=config.worker_id)
    evaluation_service_pb2_grpc.add_RolloutServicer_to_server(
        servicer, server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port', help='Port to start the service.', type=int, default=20000)
    parser.add_argument(
        '--config', help='Path to the config file.')
    parser.add_argument(
        '--worker-id', help='Worker ID.', type=int, default=0)
    parser.add_argument(
        '--run-on-gke', help='Whether run this on GKE.', default=False,
        action='store_true')
    args, _ = parser.parse_known_args()

    gin.parse_config_file(args.config)
    main(args)
