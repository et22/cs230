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

# code for dorsal net
import sys
sys.path.append('./baselines/dorsalnet/')
from paths import *
from python_dict_wrapper import wrap

import torch
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoImageProcessor, HieraModel


##### image models - resnet50s with different training procedures #####
# unt is untrained, reg is regular pretrained, adv is adversarially pretrained 
def resnet50_unt(device=None):
    model = resnet50().to(device)  
    model.eval()
    return model

def resnet50_reg(device=None):
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)    
    model.eval()
    return model

def resnet50_fcn(device=None):
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights).to(device)  
    model.eval()
    return model

def resnet50_adv(device=None):
    model = resnet50().to(device)    

    all_keys = torch.load("./checkpoints/feature_extractors/ImageNetAdv.pt", map_location=device, weights_only=False)
    # model downloaded from here - https://github.com/MadryLab/robustness_applications?

    temp = dict()
    for key in model.state_dict().keys():
        temp[key] = all_keys['model'][f'module.model.{key}']
    model.load_state_dict(OrderedDict(temp))
    model.eval()

    return model

# get feature extractor for a particular image model
def get_image_feature_extractor(layer, mod_type, device, use_pretrained):
    if mod_type == 'resnet50':
        if use_pretrained:
            model = resnet50_reg(device=device)
        else:
            model = resnet50_unt(device=device)
    if mod_type == 'resnet50_fcn':
        if not use_pretrained:
            print("Warning! Using pretrained anyway.")
            
        model = resnet50_fcn(device=device)
                    
    if mod_type == 'resnet50':
        layer_node_map = {
            'layer1':'layer1.0.bn1', 
            'layer2':'layer2.0.bn1', 
            'layer3':'layer3.0.bn1', 
            'layer4':'layer4.0.bn1', 
        }
        return_node = {layer_node_map[layer]: layer}

    if mod_type == "resnet50_fcn":
        layer_node_map = {
            'layer1':'backbone.layer1.0.bn1', 
            'layer2':'backbone.layer2.0.bn1', 
            'layer3':'backbone.layer3.0.bn1', 
            'layer4':'backbone.layer4.0.bn1', 
        }
        return_node = {layer_node_map[layer]: layer}
        
    feat_ext = create_feature_extractor(model, return_nodes=return_node)
    
    return lambda x: feat_ext(x)[layer]

##### video models - resnet3d18s with different training procedures #####
# unt is untrained, reg is regular pretrained
def resnet3d18_unt(device=None):
    model = r3d_18().to(device)  
    model.eval()
    return model

def resnet3d18_reg(device=None):
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(device)  
    model.eval()
    return model

def resnet2plus1d_unt(device=None):
    model = r2plus1d_18().to(device)  
    model.eval()
    return model

def resnet2plus1d_reg(device=None):
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1).to(device)  
    model.eval()
    return model


def dorsalnet_reg(device=None):
    features = 'airsim_04'

    args = wrap({'features': features,
                 'ckpt_root': CHECKPOINTS,
                 'slowfast_root': None,
                 'ntau': 32,
                 'nt': 1,
                 'subsample_layers': False, 
                 'device': device})
    
    from models import get_feature_model
    
    model, hooks, data = get_feature_model(args)
    model = model.to(device)
    model.eval()
    return model

def sam2t_reg(device=None):
    checkpoint = "./baselines/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml"
    model = build_sam2_video_predictor(model_cfg, checkpoint).to(device)
    model.eval()
    return model

def hierat_reg(device=None):
    model = HieraModel.from_pretrained("facebook/hiera-tiny-224-hf", output_hidden_states=True).to(device)
    model.eval()
    return model
    
# get feature extractor for a particular video model
def get_video_feature_extractor(layer, mod_type, device, use_pretrained):
    if mod_type == 'resnet3d':
        if use_pretrained:
            model = resnet3d18_reg(device=device)
        else:
            model = resnet3d18_unt(device=device)
    if mod_type == "resnet2plus1d":
        if use_pretrained:
            model = resnet2plus1d_reg(device=device)
        else:
            model = resnet2plus1d_unt(device=device)
    if mod_type == "dorsalnet":
        if not use_pretrained:
            print("Warning! Using pretrained anyway.")
        model = dorsalnet_reg(device=device)

    if mod_type == 'sam2': 
        layer_node_map = {
        'layer3': 'image_encoder',
        }
        return_nodes = {layer_node_map[layer]: layer}
        model = sam2t_reg(device=device)

    if mod_type == 'hiera':
        model = hierat_reg(device=device)

    if mod_type == 'resnet3d' or mod_type == 'resnet2plus1d':
        layer_node_map = {
            'layer1':'layer1.0.conv1.1', 
            'layer2':'layer2.0.conv1.1', 
            'layer3':'layer3.0.conv1.1', 
            'layer4':'layer4.0.conv1.1', 
        }
        return_node = {layer_node_map[layer]: layer}

    if mod_type == 'dorsalnet':
        layer_node_map = {
            'layer1': 'res0.relu',
            'layer2': 'res1.relu',
            'layer3': 'res2.relu',
            'layer4': 'res3.relu',
        }
        return_node = {layer_node_map[layer]: layer}

    if mod_type != 'sam2' and mod_type != 'hiera':
        feat_ext = create_feature_extractor(model, return_nodes=return_node)
        ret_func = lambda x: feat_ext(x)[layer]
    else:
        ret_func = lambda x: model(x[:, :, 0, :, :]).reshaped_hidden_states[-1][:, 3, 3, :].unsqueeze(1)
        
    return ret_func

# pytorch wrapper classes for MEI generation
class ImageFeatureExtractor(nn.Module):
    def __init__(self, feat_ext, stim_dims, device=torch.device('cpu')):
        super().__init__()
        self.core = feat_ext

    def initialize_stim(self):
        self.stim = nn.Parameter(torch.randn(stim_dims, requires_grad = True, dtype=torch.float32))

    def forward(self, x):
        return self.core(x)

class VideoFeatureExtractor(nn.Module):
    def __init__(self, feat_ext, stim_dims, device=torch.device('cpu')):
        super().__init__()
        self.core = feat_ext
        self.stim = nn.Parameter(torch.randn((stim_dims[0], stim_dims[1], 1, stim_dims[3], stim_dims[4]), requires_grad = True, dtype=torch.float32).repeat(1, 1, stim_dims[2], 1, 1))
    
    def initialize_stim(self):
        self.stim = nn.Parameter(torch.rand((stim_dims[0], stim_dims[1], 1, stim_dims[3], stim_dims[4]), requires_grad = True, dtype=torch.float32).repeat(1, 1, stim_dims[2], 1, 1))

    def forward(self, x):
        x = self.core(x)
        return x