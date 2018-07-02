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

import tensorflow as tf
from tensorforce.core.networks import Network


class MultiRobotLayer(Network):
    """
    End-to-end layer to control multiple robots.
    """
    def __init__(self, rate=0.0, scope='multiRobotLayer', summary_labels=()):
        self.rate = rate
        super(MultiRobotLayer, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, internals, update, return_internals=False):
        net = x['laser']
        net = tf.layers.conv1d(net, 32, 5, strides=2, activation=tf.nn.relu)
        net = tf.layers.conv1d(net, 32, 3, strides=2, activation=tf.nn.relu)
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 256, activation=tf.nn.relu)

        rel_goal = x['rel_goal']
        velocities = x['velocities']

        net = tf.concat(axis=1, values=[rel_goal, velocities, net])
        # net = tf.Print(net, [x['id'],[net]], summarize=10)
        net = tf.layers.dense(net, 256, activation=tf.nn.relu)
        net = tf.layers.dense(net, 128, activation=tf.nn.relu)
        # net = tf.layers.dense(net, 64, activation=tf.nn.relu)

        if return_internals:
            return net, {}
        else:
            return net
