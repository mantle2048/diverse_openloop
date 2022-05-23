# +
import pickle
import unittest
import pytest
from tqdm import tqdm
import numpy as np

import MetaTraj
from MetaTraj.envs.env_builder import build_regular_env
from MetaTraj.robots import a1
from MetaTraj.robots import robot_config
from MetaTraj.robots import laikago
from MetaTraj.infrastructure.utils.utils import write_gif

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# # %matplotlib inline
# -

@pytest.mark.skip(reason="env_builder is currently broken.")
class build_regular_env_test(unittest.TestCase):
    def setUp(self):
        self._env = build_regular_env(
            a1.A1,
            motor_control_mode=robot_config.MotorControlMode.POSITION,
            enable_rendering=False,
            action_limit=(0.75, 0.75, 0.75),
            on_rack=False
        )
    def test_env(self):
        print(self._env)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

env = build_regular_env(
    a1.A1,
    motor_control_mode=robot_config.MotorControlMode.POSITION,
    enable_rendering=False,
    # action_limit=0.75,
    wrap_trajectory_generator=True,
    on_rack=False
)
print(env)
print("=====" * 8)
print(env.observation_space)
print("=====" * 8)
print(env.action_space)
print("=====" * 8)
print(dir(env))
print("=====" * 8)
print(env.reward_range)
print("=====" * 8)
print(env.metadata)

env._robot.time_step

obs = env.reset()
path_return = 0.
path_length = 0
fps = env.metadata['video.frames_per_second']
FREQ = 5
image_obss = []
angle_abduction = 0.0
A_hip = 0.4
A_calf = 0.3
B = np.pi
for t in tqdm(np.linspace(0, 10, 100)):
    
    angle_hip_03 = 0.67 +  A_hip * np.sin(np.pi * FREQ * t)
    angle_hip_12 = 0.67 +  A_hip * np.sin(np.pi * FREQ * t + B)
    angle_calf_03 = -0.85 + A_calf * np.sin(np.pi * FREQ * t)
    angle_calf_12 = -0.85 + A_calf * np.sin(np.pi * FREQ * t + B)
    action = np.array([angle_abduction, angle_hip_03, angle_calf_03] + [angle_abduction, angle_hip_12, angle_calf_12] * 2 + [angle_abduction, angle_hip_03, angle_calf_03])
    # angle_abduction = -angle_abduction
    # action = np.array([angle_abduction, angle_hip, angle_calf] * 4)
    next_obs, rew, done, info = env.step(action)
    image_obs = env.render(mode='rgb_array')
    image_obss.append(image_obs)
    path_return += rew
    path_length += 1
    if done: break
print(path_length, path_return)
image_obss = np.array(image_obss)
write_gif('test_rendered_path', image_obss, fps)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
t = np.linspace(0, 10, 200)
angle_hip = 0.67 +  A_hip * np.sin(np.pi * FREQ * t)
angle_calf = -0.85 + A_calf * np.sin(np.pi * FREQ * t)
ax.plot(t,angle_hip)
ax.plot(t,angle_calf)
ax.legend(["angle_hip", "angle_calf"])
plt.show()




