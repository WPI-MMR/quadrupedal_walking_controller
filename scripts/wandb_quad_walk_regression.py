import gym
import gym_solo

from gym_solo.envs import solo8v2vanilla_realtime
from gym_solo.core import obs
from gym_solo import testing

import numpy as np
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('length', default=10, 
                    help='how many seconds to run the simulation for.')

args = parser.parse_args()


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

joints = config.starting_joint_pos.copy()
to_action = lambda d: [d[j] + config.starting_joint_pos[j] 
                        for j in env.joint_ordering]

while True:
  pos = float(input('Which position do you want to set all the joints to?: '))
  if pos == 69:
    env.reset()
    continue

  for j in joints.keys():
    joints[j] = pos
  env.step(to_action(joints))