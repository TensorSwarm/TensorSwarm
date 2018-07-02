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

import unittest

from scripts.tensorforce.ArgosEnvironment import *


class Wrap(unittest.TestCase):
    def test_case1(self):
        self.assertAlmostEqual(math.pi, wrap(math.pi))
        self.assertAlmostEqual(0.0, wrap(2 * math.pi))
        self.assertAlmostEqual(0.0, wrap(2 * math.pi))


class PoseToRelPose(unittest.TestCase):
    def checkPoseAlomstEqual(self, pose1, pose2):
        self.assertAlmostEqual(pose1.x, pose2.x)
        self.assertAlmostEqual(pose1.y, pose2.y)
        self.assertAlmostEqual(wrap(pose1.theta), wrap(pose2.theta))

    def test_zero(self):
        target = Pose2D(34, 23, 3.2)
        center = Pose2D(0, 0, 0)

        self.checkPoseAlomstEqual(target, pose_to_rel_pose(center, target))

    def test_case1(self):
        target = Pose2D(1.0, 1.0, math.pi / 2)
        center = Pose2D(2.0, 2.0, math.pi)
        goal = Pose2D(1.0, 1.0, - math.pi / 2)
        self.checkPoseAlomstEqual(goal, pose_to_rel_pose(center, target))

    def test_case2(self):
        target = Pose2D(1.0, 3.0, math.pi / 2)
        center = Pose2D(2.0, 2.0, math.pi)
        goal = Pose2D(1.0, -1.0, - math.pi / 2)
        self.checkPoseAlomstEqual(goal, pose_to_rel_pose(center, target))


class RelPolar(unittest.TestCase):
    def test_case1(self):
        target = Pose2D(1.0, 1.0, 'nan')
        center = Pose2D(2.0, 2.0, math.pi)
        dist, theta = rel_polar(center, target)
        self.assertAlmostEqual(dist, np.sqrt(2))
        self.assertAlmostEqual(wrap(theta), wrap(np.pi / 4))

    def test_case2(self):
        target = Pose2D(1.0, 3.0, 'nan')
        center = Pose2D(2.0, 2.0, math.pi)
        dist, theta = rel_polar(center, target)
        self.assertAlmostEqual(dist, np.sqrt(2))
        self.assertAlmostEqual(wrap(theta), wrap(-np.pi / 4))

    def test_case3(self):
        target = Pose2D(2.0, 3.0, 'nan')
        center = Pose2D(2.0, 2.0, math.pi / 2)
        dist, theta = rel_polar(center, target)
        self.assertAlmostEqual(dist, 1)
        self.assertAlmostEqual(wrap(theta), wrap(0.0))


if __name__ == '__main__':
    unittest.main()
