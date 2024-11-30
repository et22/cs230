import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url
from collections import OrderedDict
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.video import r3d_18, R3D_18_Weights, r2plus1d_18, R2Plus1D_18_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

import numpy as np
import pandas as pd
import torch

##### video models - resnet3d18s with different training procedures #####
# unt is untrained, reg is regular pretrained
def resnet3d18_unt(device=None):
    model = r3d_18().to(device)  
    return model

def resnet3d18_reg(device=None):
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(device)  
    return model

# get feature extractor for a particular video model
def get_video_feature_extractor(layer, mod_type, device, use_pretrained=True, freeze_weights=True):
    if use_pretrained:
        model = resnet3d18_reg(device=device)
    else:
        model = resnet3d18_unt(device=device)

    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    else:
        model.train()
    
    if mod_type == 'resnet3d':
        layer_node_map = {
            'layer1':'layer1.0.conv1.1', 
            'layer2':'layer2.0.conv1.1', 
            'layer3':'layer3.0.conv1.1', 
            'layer4':'layer4.0.conv1.1', 
        }
        return_node = {layer_node_map[layer]: 'layer'}

    feat_ext = create_feature_extractor(model, return_nodes=return_node)

    return feat_ext

# pytorch wrapper classes for synthesizing optimal stimuli 
class VideoFeatureExtractor(nn.Module):
    def __init__(self, feat_ext, stim_dims, device=torch.device('cpu')):
        super().__init__()
        self.core = feat_ext
        self.stim = nn.Parameter(torch.randn((stim_dims[0], stim_dims[1], 1, stim_dims[3], stim_dims[4]), requires_grad = True, dtype=torch.float32).repeat(1, 1, stim_dims[2], 1, 1))
    
    def initialize_stim(self):
        self.stim = nn.Parameter(torch.rand((stim_dims[0], stim_dims[1], 1, stim_dims[3], stim_dims[4]), requires_grad = True, dtype=torch.float32).repeat(1, 1, stim_dims[2], 1, 1))

    def forward(self, x):
        x = self.core(x)['layer']
        return x