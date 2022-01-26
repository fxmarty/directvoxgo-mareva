import argparse
import mmcv
import os
import torch

from lib import dvgo
from lib.utils_get_center import get_center_object
from lib.utils_interaction_3D import interaction
from lib import utils

import numpy as np

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--config',
                    type=str,
                    required=True,
                    help='path to .py config file for the experiment')

parser.add_argument("--coarse",
                    action='store_true',
                    help='Use coarse model')

parser.add_argument("--fullscreen",
                    action='store_true',
                    help='Run the visualization in full screen mode')

parser.add_argument("--init",
                    type=str,
                    required=True,
                    help='Initialization pose')

parser.add_argument("--coordinates",
                    type=str,
                    required=True,
                    help='Path to a .npz generated with the option `export_coarse_only`')

args = parser.parse_args()
cfg = mmcv.Config.fromfile(args.config)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.coarse is True:
    ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
    print('Using coarse model.')

    stepsize = cfg.coarse_model_and_render.stepsize
else:
    ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
    print('Using fine model.')

    stepsize = cfg.fine_model_and_render.stepsize

print('Loading model...')
model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)

parent_exp_folder = os.path.split(args.config)[0]
params = np.load(os.path.join(parent_exp_folder, 'params.npz'))

K = params['K']
HW = params['HW']

render_viewpoints_kwargs = {
    'model': model,
    'ndc': cfg.data.ndc,
    'render_kwargs': {
        'near': params['near'],
        'far': params['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    },
}

# load the initial pose
render_pose = np.loadtxt(args.init).astype(np.float32)
render_pose = torch.Tensor(render_pose)

# compute the coordinates for the center of the object to handle rotations later
coordinates = np.load(args.coordinates)
alpha = coordinates['alpha']
xyz_min = coordinates['xyz_min']
xyz_max = coordinates['xyz_max']

center = get_center_object(alpha, xyz_min, xyz_max)
print(f'Object center at position {center}')

interaction(HW=HW,
            render_pose=render_pose,
            K=K,
            cfg=cfg,
            fullscreen=args.fullscreen,
            center=center,
            **render_viewpoints_kwargs)
