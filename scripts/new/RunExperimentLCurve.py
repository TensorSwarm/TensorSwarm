# Copyright (C) 2018 deeplearningrobotics.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#!/usr/bin/env python3
import sys
#from baselines import logger
import ppo2
import math
from policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy, RobotPolicy
import multiprocessing
import tensorflow as tf
#import cProfile
import logger
from ArgosMultiProcessEnvironment import ArgosMultiProcessEnvironment

from geometry_msgs.msg import Twist, Pose2D
import numpy as np
def train(num_timesteps, seed):
    np.seterr(invalid='raise')

    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu,
                            )
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


    logger.configure("/home/pasa/deeplearning/tf_models/")

    tf.Session(config=config).__enter__()

    start_poses = list()

    start_poses.append(Pose2D(2.3, 1.6, math.pi/2))
    start_poses.append(Pose2D(2.3, 1.3, math.pi/2))
    # start_poses.append(Pose2D(2.3, 1.0, math.pi/2))
    # start_poses.append(Pose2D(2.7, 1.6, math.pi/2))
    # start_poses.append(Pose2D(2.7, 1.3, math.pi/2))
    # start_poses.append(Pose2D(2.7, 1.0, math.pi/2))

    start_poses.append(Pose2D(1.6, 2.3, 0.0))
    start_poses.append(Pose2D(1.3, 2.3, 0.0))
    # start_poses.append(Pose2D(1.0, 2.3, 0.0))
    # start_poses.append(Pose2D(1.6, 2.7, 0.0))
    # start_poses.append(Pose2D(1.3, 2.7, 0.0))
    # start_poses.append(Pose2D(1.0, 2.7, 0.0))

    # X neg
    start_poses.append(Pose2D(-2.3, 1.6, math.pi/2))
    start_poses.append(Pose2D(-2.3, 1.3, math.pi/2))
    # start_poses.append(Pose2D(-2.3, 1.0, math.pi/2))
    # start_poses.append(Pose2D(-2.7, 1.6, math.pi/2))
    # start_poses.append(Pose2D(-2.7, 1.3, math.pi/2))
    # start_poses.append(Pose2D(-2.7, 1.0, math.pi/2))

    start_poses.append(Pose2D(-1.6, 2.3, -math.pi))
    start_poses.append(Pose2D(-1.3, 2.3, -math.pi))
    # start_poses.append(Pose2D(-1.0, 2.3, -math.pi))
    # start_poses.append(Pose2D(-1.6, 2.7, -math.pi))
    # start_poses.append(Pose2D(-1.3, 2.7, -math.pi))
    # start_poses.append(Pose2D(-1.0, 2.7, -math.pi))

    # Y neg
    start_poses.append(Pose2D(2.3, -1.6, -math.pi/2))
    # start_poses.append(Pose2D(2.3, -1.3, -math.pi/2))
    # start_poses.append(Pose2D(2.3, -1.0, -math.pi/2))
    # start_poses.append(Pose2D(2.7, -1.6, -math.pi/2))
    # start_poses.append(Pose2D(2.7, -1.3, -math.pi/2))
    # start_poses.append(Pose2D(2.7, -1.0, -math.pi/2))
    #
    start_poses.append(Pose2D(1.6, -2.3, 0.0))
    # start_poses.append(Pose2D(1.3, -2.3, 0.0))
    # start_poses.append(Pose2D(1.0, -2.3, 0.0))
    # start_poses.append(Pose2D(1.6, -2.7, 0.0))
    # start_poses.append(Pose2D(1.3, -2.7, 0.0))
    # start_poses.append(Pose2D(1.0, -2.7, 0.0))

    # # Both neg
    start_poses.append(Pose2D(-2.3, -1.6, -math.pi/2))
    # start_poses.append(Pose2D(-2.3, -1.3, -math.pi/2))
    # start_poses.append(Pose2D(-2.3, -1.0, -math.pi/2))
    # start_poses.append(Pose2D(-2.7, -1.6, -math.pi/2))
    # start_poses.append(Pose2D(-2.7, -1.3, -math.pi/2))
    # start_poses.append(Pose2D(-2.7, -1.0, -math.pi/2))
    #
    start_poses.append(Pose2D(-1.6, -2.3, -math.pi))
    # start_poses.append(Pose2D(-1.3, -2.3, -math.pi))
    # start_poses.append(Pose2D(-1.0, -2.3, -math.pi))
    # start_poses.append(Pose2D(-1.6, -2.7, -math.pi))
    # start_poses.append(Pose2D(-1.3, -2.7, -math.pi))
    # start_poses.append(Pose2D(-1.0, -2.7, -math.pi))


######################

    goal_poses = list()

    goal_poses.append(Pose2D(0.0, 2.3, 0.0))
    goal_poses.append(Pose2D(0.3, 2.3, 0.0))
    # goal_poses.append(Pose2D(0.6, 2.3, 0.0))
    # goal_poses.append(Pose2D(0.0, 2.7, 0.0))
    # goal_poses.append(Pose2D(0.3, 2.7, 0.0))
    # goal_poses.append(Pose2D(0.6, 2.7, 0.0))

    goal_poses.append(Pose2D(2.3, 0.0, math.pi))
    goal_poses.append(Pose2D(2.3, 0.3, math.pi))
    # goal_poses.append(Pose2D(2.3, 0.6, math.pi))
    # goal_poses.append(Pose2D(2.7, 0.0, math.pi))
    # goal_poses.append(Pose2D(2.7, 0.3, math.pi))
    # goal_poses.append(Pose2D(2.7, 0.6, math.pi))

    # X neg

    goal_poses.append(Pose2D(-0.0, 2.3, 0.0))
    goal_poses.append(Pose2D(-0.3, 2.3, 0.0))
    #goal_poses.append(Pose2D(-0.6, 2.3, 0.0))
    # goal_poses.append(Pose2D(-0.0, 2.7, 0.0))
    # goal_poses.append(Pose2D(-0.3, 2.7, 0.0))
    # goal_poses.append(Pose2D(-0.6, 2.7, 0.0))

    goal_poses.append(Pose2D(-2.3, 0.0, math.pi/2))
    goal_poses.append(Pose2D(-2.3, 0.3, math.pi/2))
    # goal_poses.append(Pose2D(-2.3, 0.6, math.pi/2))
    # goal_poses.append(Pose2D(-2.7, 0.0, math.pi/2))
    # goal_poses.append(Pose2D(-2.7, 0.3, math.pi/2))
    # goal_poses.append(Pose2D(-2.7, 0.6, math.pi/2))

# Y neg

    goal_poses.append(Pose2D(0.0, -2.3, 0.0))
    #goal_poses.append(Pose2D(0.3, -2.3, 0.0))
    # goal_poses.append(Pose2D(0.6, -2.3, 0.0))
    # goal_poses.append(Pose2D(0.0, -2.7, 0.0))
    # goal_poses.append(Pose2D(0.3, -2.7, 0.0))
    # goal_poses.append(Pose2D(0.6, -2.7, 0.0))
    #
    goal_poses.append(Pose2D(2.3, -0.0, math.pi/2))
    # goal_poses.append(Pose2D(2.3, -0.3, math.pi/2))
    # goal_poses.append(Pose2D(2.3, -0.6, math.pi/2))
    # goal_poses.append(Pose2D(2.7, -0.0, math.pi/2))
    # goal_poses.append(Pose2D(2.7, -0.3, math.pi/2))
    # goal_poses.append(Pose2D(2.7, -0.6, math.pi/2))


# # Both neg
    goal_poses.append(Pose2D(-1.0, -2.3, 0.0))
#    goal_poses.append(Pose2D(-1.3, -2.3, 0.0))
#     goal_poses.append(Pose2D(-1.6, -2.3, 0.0))
#     goal_poses.append(Pose2D(-1.0, -2.7, 0.0))
#     goal_poses.append(Pose2D(-1.3, -2.7, 0.0))
#     goal_poses.append(Pose2D(-1.6, -2.7, 0.0))
#
    goal_poses.append(Pose2D(-2.3, -1.0, math.pi/2))
#    goal_poses.append(Pose2D(-2.3, -1.3, math.pi/2))
#     goal_poses.append(Pose2D(-2.3, -1.6, math.pi/2))
#     goal_poses.append(Pose2D(-2.7, -1.0, math.pi/2))
#     goal_poses.append(Pose2D(-2.7, -1.3, math.pi/2))
#     goal_poses.append(Pose2D(-2.7, -1.6, math.pi/2))


    env = ArgosMultiProcessEnvironment(start_poses, goal_poses)
    ppo2.learn(policy=RobotPolicy, env=env, nsteps=196, nminibatches=1,
               lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
               ent_coef=.01,
               lr=lambda f : f * 1.0e-3,
               cliprange=lambda f : f * 0.3,
               total_timesteps=int(num_timesteps * 1.1),
               save_interval=50,
               #restore_path="/home/pasa/deeplearning/tf_models/checkpoints/r0.3300150",
               deterministic=False
               )

def main():
    np.set_printoptions(suppress=True,
                        formatter={'float_kind':'{:0.2f}'.format})

    train(num_timesteps=30000, seed=50)

if __name__ == '__main__':
    main()

