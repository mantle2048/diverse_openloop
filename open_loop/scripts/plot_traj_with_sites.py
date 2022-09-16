import pickle
import os
import gym
import os.path as osp
import numpy as np
from open_loop.user_config import LOCAL_IMG_DIR, LOCAL_LOG_DIR
LOCAL_TARJ_DIR = osp.join(LOCAL_LOG_DIR, 'Traj')

env_name='Ant-v3' 

def load_path_and_info(idx=0, itr=200, n_path=11, resample=False):
    paths_name, paths_info_name = f'itr_{itr}_paths', f'itr_{itr}_paths_info'
    rollout_dir = osp.join(LOCAL_TARJ_DIR, f'Vae_{env_name}_{idx}', 'rollout')
    with open(osp.join(rollout_dir, paths_info_name), 'rb') as f:
        paths_info = pickle.load(f)
    with open(osp.join(rollout_dir, paths_name), 'rb') as f:
        paths = pickle.load(f)
    # paths_info = [paths_info[22], paths_info[21], paths_info[9]]
    # paths = [paths[22], paths[21], paths[9]]
    paths_info = [paths_info[20]]
    paths = [paths[20]]
    return paths, paths_info

def generate_site_txt(path_info, path_id, gap=10):
    x_position, y_position = [], []
    for info in path_info:
        x_position.append(info['x_position'])
        y_position.append(info['y_position'])
    x_position, y_position = np.array(x_position), np.array(y_position)
    x_position = x_position[range(0, len(x_position), gap)]
    y_position = y_position[range(0, len(y_position), gap)]
    site_txt_dir = osp.join(LOCAL_IMG_DIR, 'site_txt')
    os.makedirs(site_txt_dir, exist_ok=True)
    site_txt_path = osp.join(site_txt_dir, f'site_txt_{env_name}_{path_id}.txt')
    with open(site_txt_path, 'w') as f:
        for i, (x, y) in enumerate(zip(x_position, y_position)):
            main_txt = \
                f'<site name="path{path_id}_point{i}" pos="{x:.2f} {y:.2f} 0" size=".1 .1 .1" type="ellipsoid" rgba="1 0 0 1"/>\n'
            print(main_txt)
            f.write(main_txt)
    return x_position, y_position

def main():
    env = gym.make(env_name)
    env.seed(1)
    paths, paths_info = load_path_and_info()
    generate_site_txt(paths_info[0], 0)

if __name__ == '__main__':
    main()
