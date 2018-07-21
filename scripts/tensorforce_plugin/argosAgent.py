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


from tendo import singleton

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner

from scripts.new.tensorboard_logging import Logger
from ArgosEnvironment import *
from MultiRobotLayer import *

# Making sure the script only starts once.
lock = singleton.SingleInstance()

start_poses = list()
start_poses.append(Pose2D(0.0, 2.0, -math.pi/2))
start_poses.append(Pose2D(0.0, -2.0, math.pi/2))
start_poses.append(Pose2D(2.0, 0.0, math.pi))
start_poses.append(Pose2D(-2.0, 0.0, 0.0))

goal_poses = list()

goal_poses.append(Pose2D(0.0, -2.0, math.pi/2))
goal_poses.append(Pose2D(0.0, 2.0, math.pi))
goal_poses.append(Pose2D(-2.0, 0.0, math.pi))
goal_poses.append(Pose2D(2.0, 0.0, math.pi))


environment = ArgosEnvironment(start_poses, goal_poses)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

summarizer = dict(directory="/home/pasa/deeplearning/tf_board", labels=['last_rewards'])

agent = PPOAgent(
    states=environment.states,
    actions=environment.actions,
    network=MultiRobotLayer,
    summarizer=summarizer,
    # Agent
    states_preprocessing=None,
    actions_exploration=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='episodes',
        batch_size=24
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=5000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='custom',
        network=MultiRobotLayer
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=0.5e-4
        ),
        num_steps=5
    ),
    gae_lambda=0.97,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=0.5e-4
    ),
    subsampling_fraction=0.2,
    optimization_steps=25,
    execution=dict(
        type='single',
        session_config=config,
        distributed_spec=None
    ),

)

environment.set_agent(agent)

#agent.restore_model(directory="/home/pasa/deeplearning/tf_models_imp/small_batch/")
# Create the runner
runner = Runner(agent=agent, environment=environment)
# Create tensorboard logger
time_str = time.strftime("%m_%m_%H_%M_%S")
logger = Logger(log_dir="/home/pasa/deeplearning/tf_board/small_batch"+time_str)

best_reward = -100000.0
last_rewards = collections.deque(maxlen=48)
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    global last_rewards
    last_rewards.append(r.episode_rewards[-1])

    # Do not log data when the deque is not fully populated to prevent misleading results.
    if len(last_rewards) >= last_rewards.maxlen:
        step = r.episode
        mean_reward = np.mean(last_rewards)
        logger.log_scalar("last_rewards", mean_reward, step)
        logger.log_scalar("reward", r.episode_rewards[-1], step)
        logger.flush()

        if mean_reward > best_reward:
                print("Saving agent after episode {}".format(r.episode))
                r.agent.save_model('/home/pasa/deeplearning/tf_models_imp/small_batch/m'+str(mean_reward))
                global best_reward
                best_reward = mean_reward

    return True


# Start learning
runner.run(episodes=300000, max_episode_timesteps=300, episode_finished=episode_finished, deterministic=False)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
