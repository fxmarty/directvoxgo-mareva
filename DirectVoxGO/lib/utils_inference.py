import imageio
import os
import tqdm
import torch

import numpy as np

from .dvgo import get_rays_of_a_view


def render_viewpoint(model,
                     render_pose,
                     HW,
                     K,
                     ndc,
                     cfg,
                     render_kwargs,
                     render_factor=0):
    '''Render images for the given viewpoint.'''

    if render_factor != 0:
        HW = np.copy(HW)
        K = np.copy(K)
        HW //= render_factor
        K[:2, :3] //= render_factor

    c2w = render_pose

    H, W = HW
    print(H)
    print(W)

    """
    rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=not render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
    """
    #TODO understand inverse_y

    rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

    keys = ['rgb_marched']

    rays_o = rays_o.flatten(0,-2)
    rays_d = rays_d.flatten(0,-2)
    viewdirs = viewdirs.flatten(0,-2)

    print('rays_o.shape:', rays_o.shape)
    print('rays_d.shape:', rays_d.shape)
    print('viewdirs.shape:', viewdirs.shape)


    n_batch = 65536 // 16
    #n_batch = 32768
    # Needed to reduce batch size here.
    with torch.no_grad():
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(n_batch, 0),
                                  rays_d.split(n_batch, 0),
                                  viewdirs.split(n_batch, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }

    print('render_result.keys:', render_result.keys())
    rgb = render_result['rgb_marched'].cpu().numpy()

    return rgb