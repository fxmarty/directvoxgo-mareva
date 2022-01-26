import cv2
import imageio
import math
import os
import torch

import numpy as np

from lib.utils_inference import render_viewpoint
from lib import utils


def rot_phi(phi):
    res = torch.Tensor([[1, 0, 0, 0],
                        [0, np.cos(phi), -np.sin(phi), 0],
                        [0, np.sin(phi), np.cos(phi), 0],
                        [0, 0, 0, 1]]).float()
    return res
    
    
def rot_theta(th):
    res = torch.Tensor([[np.cos(th), 0, -np.sin(th), 0],
                        [0, 1, 0, 0],
                        [np.sin(th), 0, np.cos(th), 0],
                        [0, 0, 0, 1]]).float()
    return res


def rot_psi(psi):
    res = torch.Tensor([[np.cos(psi), -np.sin(psi), 0, 0],
                        [np.sin(psi), np.cos(psi), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]).float()
    return res


def translation(t):
    res = torch.eye(4)
    res[0, 3] = t[0]
    res[1, 3] = t[1]
    res[2, 3] = t[2]
    res = res.float()
    
    return res


class Pose():
    def __init__(self, pose):
        self.pose = pose
    
    def rotate(self, center, radians, angle, sign):
        self.pose = translation(- center) @ self.pose  # first move the center
        
        # perform the rotation
        if angle == 'theta':
            self.pose = rot_theta(sign * radians) @ self.pose
        elif angle == 'phi':
            self.pose = rot_phi(sign * radians) @ self.pose
        elif angle == 'psi':
            self.pose = rot_psi(sign * radians) @ self.pose
        else:
            raise ValueError('Rotation angle should be only theta, phi or psi')
    
        self.pose = translation(center) @ self.pose  # move back the center


def interaction(model,
                render_pose,
                HW,
                K,
                cfg,
                fullscreen,
                center,
                **render_viewpoints_kwargs):
    window_title = '3D Interaction'
    
    if fullscreen:
        cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_title)
        
    n_image = 0
    
    pose = Pose(render_pose)
    
    print('\nOriginal pose:')
    print(pose.pose)
    
    poses_dir = 'my_poses_jade'
    if not os.path.exists(poses_dir):
        os.makedirs(poses_dir)
    
    rad_update = math.radians(5 % 360)  # 5°
    
    keymap = {'4': 'LEFT',
              '6': 'RIGHT',
              '8': 'UP',
              '5': 'DOWN',
              '1': 'SPIN LEFT',
              '2': 'SPIN RIGHT',
              '+': 'ZOOM',
              '-': 'DEZOOM'}
    
    while True:
        code = cv2.waitKey(1) & 0xFF
        if code != 255:
            char = chr(code)
            
            if char == 'q':
                print('Pressed q: EXIT')
                break
            elif char == '4':  # left
                pose.rotate(center, radians=rad_update, angle='theta', sign=-1)
            elif char == '6':  # right
                pose.rotate(center, radians=rad_update, angle='theta', sign=1)
            elif char == '8':  # up
                pose.rotate(center, radians=rad_update, angle='psi', sign=1)
            elif char == '5':  # down
                pose.rotate(center, radians=rad_update, angle='psi', sign=-1)
            elif char == '2':  # spin right
                pose.rotate(center, radians=rad_update, angle='phi', sign=1)
            elif char == '1':  # spin left
                pose.rotate(center, radians=rad_update, angle='phi', sign=-1)
            elif char == '-':
                raise NotImplementedError()
            elif char == '+':
                raise NotImplementedError()
            
            if char in ['8', '5', '4', '6', '1', '2', '+', '-']:                
                print(f'Pressed {char}: {keymap[char]}')
                print(pose.pose)
                
                rgb = render_viewpoint(model=model,
                                       render_pose=pose.pose,
                                       HW=HW,
                                       K=K,
                                       cfg=cfg,
                                       **render_viewpoints_kwargs)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                
                with open(os.path.join(poses_dir, f'{n_image}.txt'), 'w') as f:
                    for row in pose.pose.cpu().numpy():
                        to_write = ' '.join(str(x) for x in row)
                        f.write(to_write)
                        f.write('\n')
                
                cv2.imshow(window_title, rgb)
                
                n_image +=1
    
    cv2.destroyAllWindows()