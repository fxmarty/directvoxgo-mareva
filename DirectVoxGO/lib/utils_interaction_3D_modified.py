import cv2
import imageio
import math
import os
import torch
import open3d as o3d 

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

def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

def box_3d(data):
    xyz_min = data['xyz_min'] 
    xyz_max = data['xyz_max']
    cam_lst = data['cam_lst']
    
    # Outer aabb
    aabb_01 = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 0]])
    out_bbox = o3d.geometry.LineSet()
    out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
    out_bbox.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
    out_bbox.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
    
    # Cameras
    cam_frustrm_lst = []
    for cam in cam_lst:
        cam_frustrm = o3d.geometry.LineSet()
        print("cam", cam)
        cam_frustrm.points = o3d.utility.Vector3dVector(cam)
        if len(cam) == 5:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0,0,0] for i in range(8)])
            cam_frustrm.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,2],[2,4],[4,3],[3,1]])
        elif len(cam) == 8:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0,0,0] for i in range(12)])
            cam_frustrm.lines = o3d.utility.Vector2iVector([
                [0,1],[1,3],[3,2],[2,0],
                [4,5],[5,7],[7,6],[6,4],
                [0,4],[1,5],[3,7],[2,6],
            ])
        else:
            raise NotImplementedError
        cam_frustrm_lst.append(cam_frustrm)
    
    # Show
    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=xyz_min),
        out_bbox, *cam_frustrm_lst])

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
                rend_lst= []
                rgb, rays_o, rays_d, viewdirs = render_viewpoint(model=model,
                                                                render_pose=render_pose,
                                                                HW=HW,
                                                                K=K,
                                                                cfg=cfg,
                                                                **render_viewpoints_kwargs)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rend_o = rays_o[0,0].cpu().numpy()
                rend_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
                near, far = inward_nearfar_heuristic(rend_o, ratio=0.05)
                rend_lst.append(np.array([rend_o, *(rend_o+rend_d*max(near, far*0.05))]))
                
                xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
                xyz_max = -xyz_min
                pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
                xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
                xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
                
                data = {"xyz_min" : xyz_min.cpu().numpy(), "xyz_max" : xyz_max.cpu().numpy(), "cam_lst" : np.array(rend_lst)}
                box_3d(data)
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

