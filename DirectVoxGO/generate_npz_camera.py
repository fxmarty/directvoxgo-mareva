import argparse
import glob
import mmcv
import os
import torch

from lib import dvgo
from run import load_everything

import numpy as np

def compute_bbox_by_cam_frustrm(args, cfg, H, W, K, poses, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min

    for c2w in poses:
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', required=True,
                    help='config file path')

parser.add_argument('--pose', required=True,
                    help='path to folder containing .txt poses')

parser.add_argument('--output', required=True,
                    help='path to output .npz')

args = parser.parse_args()
cfg = mmcv.Config.fromfile(args.config)

# load custom cameras
parent_exp_folder = os.path.split(args.config)[0]
params = np.load(os.path.join(parent_exp_folder, 'params.npz'))

K = params['K']
HW = params['HW']
near = params['near']
far = params['far']

H, W = HW

pose_paths = sorted(glob.glob(os.path.join(args.pose, '*txt')))

all_poses = []
for pose_path in pose_paths:
    all_poses.append(np.loadtxt(pose_path).astype(np.float32))

poses = np.stack(all_poses, 0)
poses = torch.Tensor(poses)

print('Export bbox and cameras...')
xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg,
                                               H=H,
                                               W=W,
                                               K=K,
                                               poses=poses,
                                               near=near,
                                               far=far)
cam_lst = []
#TODO careful inverse_y
for c2w in poses:
    rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
    cam_o = rays_o[0,0].cpu().numpy()
    cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
    cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))

# load original cameras (from train)
# load images / poses / camera settings / data split
data_dict = load_everything(args=args, cfg=cfg)

# export scene bbox and camera poses in 3d for debugging and visualization
poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
near, far = data_dict['near'], data_dict['far']

for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
    rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
    cam_o = rays_o[0,0].cpu().numpy()
    cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
    cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))


np.savez_compressed(args.output,
                    xyz_min=xyz_min.cpu().numpy(),
                    xyz_max=xyz_max.cpu().numpy(),
                    cam_lst=np.array(cam_lst))
print('done')