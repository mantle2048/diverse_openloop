# +
import MetaTraj
import numpy as np
from torch import nn
from MetaTraj.envs.env_builder import build_regular_env
from MetaTraj.robots import a1
from MetaTraj.robots import robot_config
from MetaTraj.utils.utils import Path, write_gif, sample_trajectory

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# %matplotlib inline
# -

class RandomPolicy(nn.Module):
    def __init__(self, env):
        self.env = env
        super().__init__()
        
    def get_action(self, obs):
        random_act = self.env.action_space.sample()
        if len(random_act.shape) > 1:
            random_act = random_act[0]
        return random_act


def video_recorder():
    env = build_regular_env(
        a1.A1,
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        enable_rendering=False,
        action_limit=(0.75, 0.75, 0.75),
        on_rack=False
    )
    fps = env.metadata['video.frames_per_second']
    random_policy = RandomPolicy(env)
    path = sample_trajectory(env, random_policy, 1000, render=True)
    image_obss = path['image_obs']
    print(image_obss[0].shape)
    write_gif("test_video_recoder", image_obss, fps)


if __name__ == '__main__':
    video_recorder()


