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
import math
import time

import numpy as np
import rospy

from tensorswarm.srv import *
from geometry_msgs.msg import Twist, Pose2D
import gym
from gym import spaces


def wrap(angle):
    """Wraps an angle into (-pi,pi].

    :param angle: The angle to wrap.
    :return: The wrapped angle in (-pi,pi].
    """
    return ((-angle + np.pi) % (2.0 * np.pi) - np.pi) * -1.0


def pose_to_rel_pose(new_center, absolute):
    """Transforms an absolute pose to one relative to a give new center.

    :param new_center: The new center the absolute pose should be relative to.
    :param absolute:  The pose to transform
    :return: The transformed pose now relative to the new center.
    """
    rel = Pose2D()
    rel.x = (absolute.x - new_center.x) * math.cos(new_center.theta) - (absolute.y - new_center.y) * math.sin(new_center.theta)
    rel.y = -(absolute.x - new_center.x) * math.sin(new_center.theta) + (absolute.y - new_center.y) * math.cos(new_center.theta)
    rel.theta = wrap(absolute.theta - new_center.theta)
    return rel


def rel_polar(center, absolute):
    """Return polar coordinates of a point relative to a given center.

    :param center: The coordinate center.
    :param absolute: The point.
    :return: Polar coordinates of the provided absolute point relative to the given center.
    """
    rel = Pose2D()
    rel.x = (absolute.x - center.x) * math.cos(-center.theta) \
            - (absolute.y - center.y) * math.sin(-center.theta)
    rel.y = -(absolute.x - center.x) * math.sin(-center.theta) \
            + (absolute.y - center.y) * math.cos(-center.theta)
    dist = math.hypot(rel.x, rel.y)
    theta = wrap(np.arctan2(rel.y, rel.x))
    return dist, theta


class ArgosEnvironment(gym.Env):
    """A tensorforce environment for the Argos robotics simulator.

    """
    def __init__(self, start_poses, goal_poses, service_name = 'AIService'):
        """The length of the start and end poses must match and determine the number of robots.

        :param start_poses: The desired start poses of the robots.
        :param goal_poses: The desired goal poses of the robots.
        """
        self.service = rospy.ServiceProxy(service_name, AIService)

        self.num_robots = len(start_poses)

        assert(len(start_poses) == len(goal_poses))
        self.start_poses = start_poses
        self.goal_poses = goal_poses

        self.prev_lasers = [None] * self.num_robots
        self.prev_prev_lasers = [None] * self.num_robots

        self.current_response = None

        self.num_envs = 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-10, high=10, shape=(512+2+2,))
        # self.observation_space = dict(laser=dict(type='float', shape=(512, 3,)),
        #                               velocities=dict(type='float', shape=(2,)),
        #                               rel_goal=dict(type='float', shape=(2,)),
        #                               id=dict(type='int', shape=(1,)))

        self.theta = [0.0]*self.num_robots
        self.dist = [0.0]*self.num_robots

    def observation_to_dict(self, observations, init=False):
        """ Extracts a single observation from a list of  observations
            from the robots and converts it to a dictionary consumable
            by the neural net.

        :param observations: A list of observations from the robots.
        :param idx: The id of the robot of which to extract the observation.
        :param init: If static data should be initalized for the very first
                     operation after a reset.
        :return: Returns a dictionary of the observations belonging to the
                 specified robot.
        """

        dict_stack = []

        if init:
            for index in range(0, self.num_robots):
                values = np.asarray([p.value for p in observations[index].laser_scan.laser_rays])
                self.prev_lasers[index] = values
                self.prev_prev_lasers[index] = values

        for idx in range(0, self.num_robots):
            ob = observations[idx]
            goal = self.goal_poses[idx]
            values = np.asarray([p.value for p in ob.laser_scan.laser_rays])

            laser = np.vstack((values,
                               self.prev_lasers[idx],
                               self.prev_prev_lasers[idx]))
            self.prev_prev_lasers[idx] = self.prev_lasers[idx]
            self.prev_lasers[idx] = values
            laser = np.swapaxes(laser, 1, 0)

            assert laser.shape == (512, 3)
            vel = np.asarray([ob.twist.linear.x, ob.twist.angular.z])

            dist, theta = rel_polar(ob.pose, goal)
            relpol = np.asarray([dist, theta])

            self.dist[idx] = dist
            self.theta[idx] = theta

            dict_stack.append({'laser': laser,
                               'velocities': vel,
                               'rel_goal': relpol,
                               'id': [idx, -idx]})

        return dict_stack
        #return relpol
        #return np.hstack((laser_stack, vel_stack, rel_goal_stack))

    def action_to_twist(self, actions):
        """ Converts and action to ROS twist.

        :param actions: The action to convert
        :return: A ROS twist
        """

        t = Twist()
        # Scaling fwd. velocity such that the robot drives forward in general.

        t.linear.x = (np.clip(actions[0], -1, 1)+1.0)/3.0
        # Scaling angular speed such that the robots rotates with more agility.
        t.angular.z = actions[1] * 2

        # Safety limiter to prevent the simulation becoming inconsistent:
        # if(abs(t.linear.x) > 3.0):
        #     print "WARNING: linear vel to large. Capping."
        #     t.linear.x = t.linear.x / abs(t.linear.x)
        # if(abs(t.angular.z) > 7.0):
        #     print "WARNING: angular vel to large. Capping."
        #     print t.angular.z
        #     t.angular.z = t.angular.z / abs(t.angular.z)*3
        #     print t.angular.z
        return t

    def step(self, actions):
        if np.isnan(actions[0][0]):
            exit()
        twists = [self.action_to_twist(action) for action in actions]
        # Comment out to test relative polar coordinates correctness
        # twists = [self.action_to_twist([self.dist[idx], self.theta[idx]]) for idx in range(0, self.num_robots)]
        service_success = False
        service_exception = False

        while not service_success:
            try:
                response = self.service(twists, [], [])
                service_success = True
                if service_exception:
                    # If we got an exception we abort the trajectory, as the simulation might be in an inconstant state.
                    response.done = [True] * self.num_robots
            except Exception as e:
                print "Serice call failed. Trying again in 3 seconds: " + str(e)
                service_exception = True
                time.sleep(3)

        self.current_response = response

        return range(0, self.num_robots), self.observation_to_dict(response.observations), \
               np.asarray(response.rewards), \
               np.asarray(response.done), \
               []

    def reset(self):
        t = Twist()
        twists = list()
        twists.append(t)
        service_success = False
        while not service_success:
            try:
                response = self.service(list(), self.start_poses, self.goal_poses)
                service_success = True
            except Exception as e:
                print "Service call failed. Trying again in 3 seconds: " + str(e)
                time.sleep(3)

        self.current_response = response

        return range(0, self.num_robots), self.observation_to_dict(response.observations, True)
