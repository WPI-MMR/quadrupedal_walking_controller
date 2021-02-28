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
import wandb


class DistanceReward(rewards.Reward):
  def __init__(self, robot_id):
    self.robot = robot_id

  def compute(self) -> float:
    (x, y, _), _ = self.client.getBasePositionAndOrientation(self.robot)
    return math.sqrt(x ** 2 + y ** 2)

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
  joints['FL_HFE'] = 1.1 * -value
  joints['HR_HFE'] = value


def FLHR_KFE(joints, value):
  joints['FL_KFE'] = value
  joints['HR_KFE'] = -value


def FRHL_HFE(joints, value):
  joints['FR_HFE'] = 1.1 * value
  joints['HL_HFE'] = -value


def FRHL_KFE(joints, value):
  joints['FR_KFE'] = -value
  joints['HL_KFE'] = -value
  

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--length', default=10, type=int,
                    help='how many seconds to run the simulation for.')
parser.add_argument('-dt', '--reward_dt', default=.05, type=int, dest='dt',
                    help='how often to sample the reward.')

args, unknown = parser.parse_known_args()
for arg in unknown:
  if arg.startswith(("-", "--")):
      parser.add_argument(arg.split('=')[0])
args = parser.parse_args()

# Reward configuration
args.flat_reward_hard_margin = 0
args.flat_reward_soft_margin = 0.3
args.height_reward_target = 0.2
args.height_reward_hard_margin = 0.005
args.height_reward_soft_margin = 0.15
args.speed_reward_target = .25
args.speed_reward_hard_margin = 0
args.speed_reward_soft_margin = 0.1

# Trot configuration
args.trot_hip_launch = -0.8
args.trot_knee_launch = 1.4
args.trot_launch_dur = 0.25
args.trot_knee_clearance = 2
args.trot_clearance_dur = 0.1
args.trot_hip_step = -0.05
args.trot_knee_step = 1.5
args.trot_step_dur = 0.1


# wandb.init(
#   project='quadrupedal-walking',
#   entity='wpi-mmr',
#   config=args,
#   tags=['multiplicative_reward'],
# )


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
walk_reward = rewards.MultiplicitiveReward(1, flat_reward, height_reward, 
                                           speed_reward)
walk_reward.client = env.client

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
time.sleep(2)

end = time.time() + args.length
scorer = threading.Thread(target=episode_listener)
scorer.start()

while time.time() < end:
  # Get ready to launch FR and HL
  FLHR_HFE(joints, -1)
  FLHR_KFE(joints, 1.2)
  FRHL_KFE(joints, 1.8)
  env.step(to_action(joints))
  time.sleep(args.trot_launch_dur)

  # Make the FR and HL Movement
  FRHL_HFE(joints, -.4)
  FRHL_KFE(joints, 1.5)
  env.step(to_action(joints))
  time.sleep(args.trot_step_dur)

  # Get ready to launch FL and HR
  FRHL_HFE(joints, -1)
  FRHL_KFE(joints, 1.2)
  FLHR_KFE(joints, 1.8)
  env.step(to_action(joints))
  time.sleep(args.trot_launch_dur)

  # Make the FL and HR Movement
  FLHR_HFE(joints, -.4)
  FLHR_KFE(joints, 1.5)
  env.step(to_action(joints))
  time.sleep(args.trot_step_dur)

scorer.join()
env.close()

scores = np.array(epi_rewards)
print(f'Average Score: {np.array(epi_rewards).mean()}')
print(f'Cum Score: {np.array(epi_rewards).sum()}')

plt.plot(epi_times, epi_rewards)
plt.title('Reward over Episode')
plt.xlabel('Simulation Time (seconds)')
plt.ylabel('Rewards')

wandb.log({
  'mean_reward': scores.mean(),
  'cum_reward': scores.sum(),
  'rewards_vs_time': wandb.Image(plt)
})