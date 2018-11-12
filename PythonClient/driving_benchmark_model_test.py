#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This script is modified for testing trained model.
"""
import argparse
import logging
from IPython.core.debugger import Tracer
from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import CoRL2017
from carla.driving_benchmark.experiment_suites import BasicExperimentSuite

from old_utils import ModelControl
from carla.agent import Agent

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        dest='verbose',
        help='print some extra status information')
    argparser.add_argument(
        '-db', '--debug',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town02',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='town02_test',
        help='The name of the log file to be created by the benchmark'
    )
    argparser.add_argument(
        '--corl-2017',
        action='store_true',
        default=True,
        help='If you want to benchmark the corl-2017 instead of the Basic one'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        default=False,
        help='If you want to continue the experiment with the same name'
    )


    # Model related info
    argparser.add_argument(
        '--net',
        default="/home/li/carla-0.8.2/trained_model/relu_act_new_whole_model0626.py",
        type=str)

    argparser.add_argument(
        '--model_weights',
        default="/home/li/carla-0.8.2/trained_model/trained_model.h5",
        type=str)

    argparser.add_argument(
        '--test_benchmark',
        action='store_true',
        default=True)

    argparser.add_argument("--rgb_shape", default=(88,200,3))
    argparser.add_argument("--depth_shape", default=(88,200,1))
    argparser.add_argument("--classes", default=13, type=int)
    argparser.add_argument("--monocular_rgb", default=True)
    argparser.add_argument("--strides", default=8, type=int)
    argparser.add_argument("--w_pad", default=2, type=int)
    argparser.add_argument("--predict_view_save", default=False)
    argparser.add_argument("--input_speed", default=False)
    argparser.add_argument("--save_input", default=False)

    args = argparser.parse_args()


    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    # Load model
    agent = ModelControl(args)
    agent.load_model(args)

    # We instantiate an experiment suite. Basically a set of experiments
    # that are going to be evaluated on this benchmark.
    if args.corl_2017:
        experiment_suite = CoRL2017(args.city_name)
    else:
        print (' WARNING: running the basic driving benchmark, to run for CoRL 2017'
               ' experiment suites, you should run'
               ' python driving_benchmark_example.py --corl-2017')
        experiment_suite = BasicExperimentSuite(args.city_name)

    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, experiment_suite, args.city_name,
                          args.log_name, args.continue_experiment,
                          args.host, args.port)
