import cv2
import imageio
import math
import os
import torch

import numpy as np

from lib.utils_inference import render_viewpoint
from lib import utils


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()
imageio

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

theta, phi, radius = 0.0, 0.0, 1.


def interaction(model,
                render_pose,
                HW,
                K,
                cfg,
                fullscreen,
                **render_viewpoints_kwargs):
    window_title = '3D Interaction'
    
    print('before fullscreen')
    if fullscreen:
        cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_title)

    quit_now = False
    theta, phi, radius = 0.0, 0.0, 1
    
    print('here now')
    update = False
    
    n_image = 0
    
    poses_dir = 'my_poses_jade'
    if not os.path.exists(poses_dir):
        os.makedirs(poses_dir)
        
    while True:
        code = cv2.waitKey(1) & 0xFF
        if code != 255:
            char = chr(code)
            if char == 'q':
                print('Pressed q: exit')
                quit_now = True
            
            elif char == '4':  # left
                print('Pressed 4: left')
                theta = 5
                theta_rad = math.radians(theta%360)
                rotation = rot_theta(- theta_rad)
                render_pose = rotation @ render_pose
                update = True
            
            elif char == '6':  # right
                print('Pressed 6: right')
                theta = 5
                theta_rad = math.radians(theta%360)
                rotation = rot_theta(theta_rad)
                render_pose = rotation @ render_pose
                update = True
                
            elif char == '8':  # up
                print('Pressed 8: up')
                phi = 5
                phi_rad = math.radians(phi%360)
                rotation = rot_phi(phi_rad)
                
                render_pose = rotation @ render_pose
                update = True
            
            elif char == '5':  # down
                print('Pressed 5: down')
                phi = 5
                phi_rad = math.radians(phi%360)
                rotation = rot_phi(- phi_rad)
                
                render_pose = rotation @ render_pose
                update = True
                
            elif char == '-':
                raise NotImplementedError()
                print('Pressed -: dezoom')
                radius += 1.
                radius = min(radius, 20.)
                update = True
            
            elif char == '+':
                raise NotImplementedError()
                print('Pressed +: zoom')
                radius -= 1.
                radius = max(radius, 0.1)
                update = True
            
            if update is True:
                #render_pose = pose_spherical(theta, phi, radius)
                #pose_path = '/home/felix/Documents/Mines/3A/Option/Mini-projet/directvoxgo-mareva/DirectVoxGO/data/BlendedMVS/Jade/pose/1_0000_00000011.txt'
                #render_pose = np.loadtxt(pose_path).astype(np.float32)
                
                #render_pose = torch.Tensor(render_pose)
                
                print(render_pose)
                
                rgb = render_viewpoint(model=model,
                                       render_pose=render_pose,
                                       HW=HW,
                                       K=K,
                                       cfg=cfg,
                                       **render_viewpoints_kwargs)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                update = False
                
                #with open('rgb.npy', 'wb') as f:
                #    np.save(f, rgb)
                
                #rgb8 = utils.to8b(rgb)
                #filename = f'teeest_{n_image}.jpg'
                #imageio.imwrite(filename, rgb8)
                
                with open(os.path.join(poses_dir, f'{n_image}.txt'), 'w') as f:
                    for row in render_pose.cpu().numpy():
                        to_write = ' '.join(str(x) for x in row)
                        f.write(to_write)
                        f.write('\n')
                
                cv2.imshow(window_title, rgb)
                
                n_image +=1
    
        if quit_now:
            break
        
    cv2.destroyAllWindows()

