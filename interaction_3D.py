# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:31:41 2022

@author: ASUS
"""

import cv2
import math

from inference import render_viewoint

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


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

theta, phi, radius = 0.0, 0.0, 1.


def Interaction(fullscreen = False)
    window_title = '3D Interaction'
    if fullscreen:
        cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_title)

    quit = False
    theta, phi, radius = 0.0, 0.0, 1.
    
    whilt True:
        code = cv2.waitKey(1) & 0xFF
        if code != 255:
                char = chr(code)
                
                if char == 'q':
                    quit = True
                
                elif char == 'w':
                    theta += 5.0 #en degre
                    theta_rad = math.radians(theta%360)
                    
                    # translation + rotation TODO
                    
                elif char == 's':
                    theta -= 5.0
                    theta_rad = math.radians(theta%360)
                    
                elif char == 'a':
                    phi -= 5.0
                    phi_rad = math.radians(phi%360)
                
                elif char == 'd':
                    phi += 5.0
                    phi_rad = math.radians(phi%360)
                    
                elif char == 'g':
                    radius += 1.
                    radius = min(radius, 20.)
                
                elif char == 'p':
                    radius -= 1.
                    radius =max(radius, 0.1)
                
                render_poses = pose_spherical(theta, phi, radius)
                
                rgb = render_viewoint(model, render_poses) # pose is 4*4 matrix           
        
        cv2.imshow(window_title, rgb)
    
        if quit:
            break
        
    cv2.destroyAllWindows()

