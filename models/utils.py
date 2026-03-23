import math

import torch.nn as nn
import torch.nn.functional as F



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.InstanceNorm3d):
        if m.weight is not None:
            nn.init.normal_(m.weight, mean=1, std=0.02)
        if m.bias is not None: 
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv3d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose3d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


def resize_pos_embed(posemb, grid_new_shape):
    posemb_grid = posemb[0, :]

    gs_h, gs_w, gs_d = grid_new_shape
    
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w, gs_d), mode="trilinear", align_corners=False)
    
    posemb_grid = posemb_grid.reshape(1, -1, gs_h, gs_w, gs_d)
    
    return posemb_grid
