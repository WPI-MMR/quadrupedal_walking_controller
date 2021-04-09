import rclpy
from rclpy.node import Node

from trajectory_interfaces import msg

import time
import numpy as np


class Trot(Node):
  def __init__(self) -> None:
    super().__init__('trot')

    self.joints = {
      'FL_HFE': 0,
      'FL_KFE': 0,
      'FL_ANKLE': 0,
      'FR_HFE': 0,
      'FR_KFE': 0,
      'FR_ANKLE': 0,
      'HL_HFE': 0,
      'HL_KFE': 0,
      'HL_ANKLE': 0,
      'HR_HFE': 0,
      'HR_KFE': 0,
      'HR_ANKLE': 0,
    }

    # Create publisher
    self.joint_publisher = self.create_publisher(
      msg.JointAngles, 'joint_angles', 10)

    # Create to ensure that the connection is up with com_serial. Not actually 
    # used.
    self.sensors_client = self.create_create(
      msg.SensorDataRequest, 'sensor_data_request')

    while not self.sensors_client.wait_for_service(timeout_sec=1.):
      self.get_logger().warning('Serial service inactive, waiting...')

  def send_angles(self):
    joint_msg = msg.JointAngles()
    
    joint_msg.left_hip = self.joints['HL_HFE']
    joint_msg.left_knee = self.joints['HL_KFE']
    joint_msg.right_hip = self.joints['HR_HFE']
    joint_msg.right_knee = self.joints['HR_KFE']
    joint_msg.left_shoulder = self.joints['FL_HFE']
    joint_msg.left_elbow = self.joints['FL_KFE']
    joint_msg.right_shoulder = self.joints['FR_HFE']
    joint_msg.right_elbow = self.joints['FR_KFE']

    self.get_logger().debug('Sending the following robot configuration: {}'.format(self.joints))
    self.joint_publisher.publish(joint_msg)

  def FLHR_HFE(self, value):
    self.joints['FL_HFE'] = -value
    self.joints['HR_HFE'] = value

  def FLHR_KFE(self, value):
    self.joints['FL_KFE'] = value
    self.joints['HR_KFE'] = -value

  def FRHL_HFE(self, value):
    self.joints['FR_HFE'] = value
    self.joints['HL_HFE'] = -value

  def FRHL_KFE(self, value):
    self.joints['FR_KFE'] = -value
    self.joints['HL_KFE'] = -value


def trot(args=None):
  """Run a super rudamentary, hardcoded trot in quadrupedal mode"""
  rclpy.init(args=args)
  trot = Trot()

  # Initialize robot position
  trot.joints = {
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
    'HR_ANKLE': 0,
  }
  trot.send_angles()

  """
  # Robot configuration values
  trot_hip_launch = -0.8
  trot_knee_launch = 1.4
  trot_launch_dur = 0.25
  trot_knee_clearance = 2
  trot_clearance_dur = 0.1
  trot_hip_step = -0.05
  trot_knee_step = 1.5
  trot_step_dur = 0.1

  # while True:
  # Get ready to launch FR and HL
  trot.FLHR_HFE(trot_hip_launch)
  trot.FLHR_KFE(trot_knee_launch)
  trot.FRHL_KFE(trot_knee_clearance)
  trot.send_angles()
  time.sleep(trot_launch_dur)

  # Make the FR and HL Movement
  trot.FRHL_HFE(trot_hip_step)
  trot.FRHL_KFE(trot_knee_step)
  trot.send_angles()
  time.sleep(trot_step_dur)

  # Get ready to launch FL and HR
  trot.FRHL_HFE(trot_hip_launch)
  trot.FRHL_KFE(trot_knee_launch)
  trot.FLHR_KFE(trot_knee_clearance)
  trot.send_angles()
  time.sleep(trot_launch_dur)

  # Make the FL and HR Movement
  trot.FLHR_HFE(trot_hip_step)
  trot.FLHR_KFE(trot_knee_step)
  trot.send_angles()
  time.sleep(trot_step_dur)
  """
  
  trot.destroy_node()
  rclpy.shutdown()