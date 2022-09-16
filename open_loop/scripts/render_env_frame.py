import gym
import os
import os.path as osp
import numpy as np
import pybullet_envs
from PIL import Image
from open_loop.user_config import LOCAL_IMG_DIR, LOCAL_RENDER_CONFIG
from pyvirtualdisplay import Display

ENV = (
    "Ant-v3",
    # "HalfCheetah-v3",
    # "MinitaurBulletEnv-v0",
    # "MinitaurReactiveEnv-v0",
)

def get_frame(env, frame_num = 1):
    assert frame_num >= 1
    obs = env.reset()
    frames = [env.render(**LOCAL_RENDER_CONFIG)]
    for _ in range(frame_num - 1):
        act = np.zeros(env.action_space.sample().shape)
        env.step(act)
        frames.append(env.render(**LOCAL_RENDER_CONFIG))
    return frames

def save_frame(env_name, frames):
    frame_dir = osp.join(LOCAL_IMG_DIR, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame_name = f'{env_name}_{idx}.png'
        frame_path = osp.join(frame_dir, frame_name)
        im = Image.fromarray(frame)
        im.save(frame_path)
        print(f"Saving {frame_name} .......")

def main():
    virtual_disp = Display(visible=False, size=(1400, 900))
    virtual_disp.start()
    for env_name in ENV:
        env = gym.make(env_name)
        env.seed(1)
        frames = get_frame(env, frame_num=10)
        save_frame(env_name, frames)

if __name__ == '__main__':
    main()
