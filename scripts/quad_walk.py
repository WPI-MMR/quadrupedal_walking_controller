import gym
import gym_solo

from gym_solo.envs import solo8v2vanilla_realtime
from gym_solo.core import rewards
from gym_solo import testing

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import threading
import time


epi_times, epi_rewards = [], []
def episode_listener():
  global epi_times
  global epi_rewards
  global end
  curr_timestep = 0.
  with tqdm(total=int(0.9 * math.floor((end - time.time())/args.dt)),
            desc='Evaluating Episode') as t:
    while time.time() < end:
      epi_times.append(curr_timestep)
      epi_rewards.append(walk_reward.compute())
      curr_timestep += args.dt
      t.update()
      time.sleep(args.dt)
  return epi_times, epi_rewards


def FLHR_HFE(joints, value):
  joints['FL_HFE'] = -value
  joints['HR_HFE'] = value


def FLHR_KFE(joints, value):
  joints['FL_KFE'] = value
  joints['HR_KFE'] = -value


def FRHL_HFE(joints, value):
  joints['FR_HFE'] = value
  joints['HL_HFE'] = -value


def FRHL_KFE(joints, value):
  joints['FR_KFE'] = -value
  joints['HL_KFE'] = -value


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--length', default=10, type=int,
                    help='how many seconds to run the simulation for.')
parser.add_argument('-dt', '--reward_dt', default=.05, type=int, dest='dt',
                    help='how often to sample the reward.')

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

to_action = lambda d: [d[j] + config.starting_joint_pos[j] 
                        for j in env.joint_ordering]
joints = {
  'FL_HFE': np.pi / 4,
  'FL_KFE': np.pi / 2,
  'FL_ANKLE': 0,
  'FR_HFE': -np.pi / 4,
  'FR_KFE': -np.pi / 2,
  'FR_ANKLE': 0,
  'HL_HFE': np.pi / 4,
  'HL_KFE': -np.pi / 2,
  'HL_ANKLE': 0,
  'HR_HFE': -np.pi / 4,
  'HR_KFE': -np.pi / 2,
  'HR_ANKLE': 0
}
env.step(to_action(joints))
time.sleep(0.25)

end = time.time() + args.length
scorer = threading.Thread(target=episode_listener)
scorer.start()

FLHR_KFE(joints, 1.2)
while time.time() < end:
  # Get ready to launch FR and HL
  FLHR_HFE(joints, -0.917)
  env.step(to_action(joints))
  time.sleep(0.25)

  # Move FR and HL foot up so it can step
  FRHL_KFE(joints, 2)
  env.step(to_action(joints))
  time.sleep(0.1)

  # Make the FR and HL Movement
  FRHL_HFE(joints, -0.1)
  FRHL_KFE(joints, 1.2)
  env.step(to_action(joints))
  time.sleep(0.1)

  # Get ready to launch FL and HR
  FRHL_HFE(joints, -0.917)
  env.step(to_action(joints))
  time.sleep(0.25)

  # Move FL and HR foot up so it can step
  FLHR_KFE(joints, 2)
  env.step(to_action(joints))
  time.sleep(0.1)

  # Make the FL and HR Movement
  FLHR_HFE(joints, -0.1)
  FLHR_KFE(joints, 1.2)
  env.step(to_action(joints))
  time.sleep(0.1)


scorer.join()
print(f'Average Score: {np.array(epi_rewards).mean()}')

plt.plot(epi_times, epi_rewards)
plt.title('Reward over Episode')
plt.xlabel('Simulation Time (seconds)')
plt.ylabel('Rewards')
plt.show()