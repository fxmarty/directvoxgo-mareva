import argparse
import mmcv
import os
import torch

from lib import dvgo
from lib import utils

from lib.utils_inference import render_viewpoint


parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--config',
                    required=True,
                    help='path to .py config file for the experiment')

parser.add_argument("--model",
                    type=str,
                    default=None,
                    help='specific weights npy file to reload for coarse network')

parser.add_argument("--coarse",
                    action='store_true',
                    help='Use coarse model')


args = parser.parse_args()
cfg = mmcv.Config.fromfile(args.config)

# init enviroment
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

render_viewpoints_kwargs = {
    'model': model,
    'ndc': cfg.data.ndc,
    'render_kwargs': {
        'near': cfg.misc['near'],
        'far': cfg.misc['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    },
}

print('WARNING: all training images assumed to be of same shape')

#TODO define
# render_pose =
K = cfg.misc['K']
HW = cfg.misc['HW']


rgb = render_viewpoint(render_pose=render_pose,
                       HW=HW,
                       K=K,
                       **render_viewpoints_kwargs)