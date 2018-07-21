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
from tensorforce.environments import Environment



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


class ArgosEnvironment(Environment):
    """A tensorforce environment for the Argos robotics simulator.

    """
    def __init__(self, start_poses, goal_poses):
        """The length of the start and end poses must match and determine the number of robots.

        :param start_poses: The desired start poses of the robots.
        :param goal_poses: The desired goal poses of the robots.
        """
        self.service = rospy.ServiceProxy('AIService', AIService)

        self.num_robots = len(start_poses)

        assert(len(start_poses) == len(goal_poses))
        self.start_poses = start_poses
        self.goal_poses = goal_poses

        self.prev_lasers = [None] * self.num_robots
        self.prev_prev_lasers = [None] * self.num_robots

        self.current_response = None
        self.agent = None

        self.id = 0

    @property
    def states(self):
        print "getting states"
        return dict(laser=dict(type='float', shape=(512, 3,)),
                    velocities=dict(type='float', shape=(2,)),
                    rel_goal=dict(type='float', shape=(2,)),
                    id=dict(type='int', shape=(1,)))

    @property
    def actions(self):
        return dict(type='float', shape=2, min_value=-1.0, max_value=1.0)

    def observation_to_dict(self, observations, idx, init=False):
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
        if init:
            for index in range(0, self.num_robots):
                values = np.asarray([p.value for p in observations[index].laser_scan.laser_rays])
                self.prev_lasers[index] = values
                self.prev_prev_lasers[index] = values

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

        return {'laser': laser, 'velocities': vel, 'rel_goal': relpol, 'id': np.asarray([idx])}

    def set_agent(self, agent):
        """ Sets the tensorforce agent the non-learning robots derive their actions from.

        :param agent:
        """
        self.agent = agent

    def action_to_twist(self, actions):
        """ Converts and action to ROS twist.

        :param actions: The action to convert
        :return: A ROS twist
        """
        t = Twist()
        # Scaling fwd. velocity such that the robot drives forward in general.
        t.linear.x = (actions[0]+0.8)/2.0
        # Scaling angular speed such that the robots rotates with more agility.
        t.angular.z = actions[1] * 5
        return t

    def execute(self, actions):

        twists = [None] * self.num_robots
        twists[self.id] = self.action_to_twist(actions)

        for idx in range(0, self.num_robots):
            if idx != self.id:
                # Only one agent learns, the others acts on the learning of the other.
                t2 = self.agent.act(self.observation_to_dict(self.current_response.observations, idx),
                                    independent=True, deterministic=True)
                twists[idx] = self.action_to_twist(t2)

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
        return self.observation_to_dict(response.observations, self.id), response.done[self.id], \
               response.rewards[self.id]

    def reset(self):
        self.id = (self.id + 1) % self.num_robots
        print "Training ID: " + str(self.id)

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

        return self.observation_to_dict(response.observations, self.id, True)
