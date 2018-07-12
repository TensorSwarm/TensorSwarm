# TensorSwarm: A framework for reinforcement learning of robot swarms.

# Introduction

*This framework was developed in private time and just released. You WILL run into problems when trying to use it. Please create issues for problems you encounter and I will do my best to
resolve them*.

TensorSwarm is an open source framework for reinforcement learning of robot swarms. Each robot in the swarm uses the same single behaviour policy with the included neural network,
but different approaches are also supported.

It connects the open source robot swarm simulation [Argos 3](https://github.com/ilpincy/argos3) to either
[Open AIs PPO](https://github.com/openai/baselines) or [TensorForce](https://github.com/reinforceio/tensorforce) using [ROS](http://www.ros.org/).

A working neural network configuration is already provided following the ideas from [Towards Optimally Decentralized Multi-Robot Collision Avoidance via
Deep Reinforcement Learning](https://arxiv.org/abs/1709.10082)

A video of the framwork in action can be found here: https://www.youtube.com/watch?v=GDV5NgrER5U

# Features

* Training up to 100 robots in multiple simulation environments.
* A ROS service handles all the communication between the simulation and the learning environment. This service is easily extendable to new sensor or command data.
* Support of multiple simulation environments at the same time to optimally use multicore CPUs.
* Contains examples which can be trained out of the box and on which users can build their extensions.

# Installation

## Dependencies

### Argos 3

Argos 3 must be installed from [here](https://github.com/deeplearningrobotics/argos3/commits/adding_lidar_clean). This version extends the simulation by a laser scanner which is missing in the
official version.

### ROS

ROS Kinetic is required which can be found [here](http://wiki.ros.org/kinetic/Installation)

### Python Dependencies

Note that ROS does not support Python 3, so all python dependencies must be installed for Python 2.

`pip install tensorflow-gpu`

If you want to use the Tensorforce integration you also need to install it as described [here](https://github.com/reinforceio/tensorforce)

## TensorSwarm

TensorSwarm comes as a ROS packages. So all you need to do is clone it into your catkin workspace.

# Running

Running TensorSwarm consits of 4 steps:

1. Setting up the environment.
2. Starting `roscore`
3. Starting the Argos 3 simulation instances.
4. Starting the Python script for generating actions and learning.

We include two experiments:

1. 8 Robots crossing in an 4-way-crossing.
2. 8 Robots crossing in a L-curve.

## Variants

The are two variants of TensorSwarm:

### OpenAI PPO

This is a true multi agent PPO implementation which should be used by advanced users. We trained swarms of over to 100 robots successfully with it.

Note that you most likely need to start with less robots, train them and then add more robots after the models.

To run this experiments execute in your catkin work space:
1. 4-way-crossing: `src/tensorswarm/run/4_way_run_ppo.bash`
2. L-curve: `src/tensorswarm/run/l_curve_run_ppo.bash`


## TensorForce

As TensorForce does not support multiple agents at the moment this variant only trains one agent at a time. Therefore this variant highly sample inefficent.
But it is well suitable for less experienced users or users who want to experiment with different deep learning algorithms other than PPO. We trained up to 4 robots
successfully with this variant.
To run this experiments execute in your catkin work space:
1. 4-way-crossing: `src/tensorswarm/run/4_way_run_tforce.bash`

## Starting the simulation

For each simulation window you have to hit FastForward (>> Button) to get the simulations running as they all start paused by default.

# Contact

For personal request or commercial support write to: ap@deeplearningrobotics.com









