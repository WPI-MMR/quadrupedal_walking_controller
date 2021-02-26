import gym
import gym_solo

from gym_solo.envs import solo8v2vanilla_realtime
from gym_solo.core import rewards
from gym_solo import testing

import numpy as np
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--length', default=10, type=int,
                    help='how many seconds to run the simulation for.')

args = parser.parse_args()
args.flat_reward_hard_margin = 0.05
args.flat_reward_soft_margin = .5
args.height_reward_target = 0.2
args.height_reward_hard_margin = 0.01
args.height_reward_soft_margin = 0.15
args.speed_reward_target = .25
args.speed_reward_hard_margin = 0.05
args.speed_reward_soft_margin = 0.1


config = solo8v2vanilla_realtime.RealtimeSolo8VanillaConfig()
config.urdf_path = 'assets/solo8_URDF_v2/solo8_URDF_v2.urdf'

# Set the robot to quadrupedal standing
config.starting_joint_pos = {
  'FL_HFE': -np.pi / 4,
  'FL_KFE': -np.pi / 2,
  'FL_ANKLE': 0,
  'FR_HFE': np.pi / 4,
  'FR_KFE': np.pi / 2,
  'FR_ANKLE': 0,
  'HL_HFE': -np.pi / 4,
  'HL_KFE': np.pi / 2,
  'HL_ANKLE': np.pi / 2,
  'HR_HFE': np.pi / 4,
  'HR_KFE': np.pi / 2,
  'HR_ANKLE': np.pi / 2
}

env = gym.make('solo8vanilla-realtime-v0', config=config)
env.obs_factory.register_observation(testing.CompliantObs(env.robot))

flat_reward = rewards.FlatTorsoReward(
  env.robot, args.flat_reward_hard_margin, args.flat_reward_soft_margin)
height_reward = rewards.TorsoHeightReward(
  env.robot, args.height_reward_target, args.height_reward_hard_margin,
  args.height_reward_soft_margin)
speed_reward = rewards.HorizontalMoveSpeedReward(
  env.robot, args.speed_reward_target, hard_margin=args.speed_reward_hard_margin,
  soft_margin=args.speed_reward_soft_margin)

walk_reward = rewards.AdditiveReward()
walk_reward.client = env.client
walk_reward.add_term(1, flat_reward)
walk_reward.add_term(1, height_reward)
walk_reward.add_term(1, speed_reward)

joints = config.starting_joint_pos.copy()
to_action = lambda d: [d[j] + config.starting_joint_pos[j] 
                        for j in env.joint_ordering]

# end = time.time() + args.length
# while time.time() < end:
print(env.joint_ordering)
env.step([0] * len(env.joint_ordering))
while True:
  print('flat: {:.4f} height: {:.4f} speed: {:.4f} overall: {:.4f}'.format(
    flat_reward.compute(), height_reward.compute(), speed_reward.compute(), 
    walk_reward.compute()))