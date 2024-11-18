import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from fix_models.feature_extractors import resnet3d18_reg, resnet3d18_unt, get_video_feature_extractor, resnet50_reg, resnet50_unt, get_image_feature_extractor, ImageFeatureExtractor, VideoFeatureExtractor, resnet2plus1d_reg, resnet2plus1d_unt

from fix_models.readouts import PoissonGaussianReadout, TransformerReadout

## pytorch full model
class BaselineModel(nn.Module):
    def __init__(self, modality, train_dataset, device=torch.device("cpu")):
        super().__init__()

        if modality == "video":
            feat_ext = get_video_feature_extractor(layer=layer, mod_type=feat_ext_type, device=device, use_pretrained=use_pretrained)
            feat_ext = VideoFeatureExtractor(feat_ext, stim_shape, device=device)
        elif modality == "image":
            feat_ext = get_image_feature_extractor(layer=layer, mod_type=feat_ext_type, device=device, use_pretrained=use_pretrained)
            feat_ext = ImageFeatureExtractor(feat_ext, stim_shape, device=device)

        readout_input = feat_ext(train_dataset[0][0].unsqueeze(0).to(device))
        print(f"readout input shape: {readout_input.shape}")
        if modality == "video" and flatten_time:
            num_input  = readout_input.shape[1] * readout_input.shape[2]
        else:
            num_input = readout_input.shape[1]
            
        num_neurons = len(train_dataset[0][1])

        for param in feat_ext.parameters():
            param.requires_grad = False

        if feat_ext_type == 'hiera':
            readout = TransformerReadout(num_input=num_input, num_neurons=num_neurons, device=device)
        else:
            readout = PoissonGaussianReadout(num_input, num_neurons, use_sigma = use_sigma, use_pool = use_pool, pool_size = pool_size, pool_stride= pool_stride, center_readout = center_readout, modality=modality, flatten_time=flatten_time, mlp=mlp, device=device)

        self.model = nn.Sequential(
            feat_ext,
            readout
        )
    def forward(self, x):
        return self.model(x)
        
class FullModel(nn.Module):
    def __init__(self, modality, layer, stim_shape, train_dataset, feat_ext_type = 'resnet50', use_sigma = True, use_pool = False, pool_size = 2, pool_stride = 2, center_readout=False, use_pretrained = True, flatten_time = False, mlp=False, device=torch.device("cpu")):
        super().__init__()

        if modality == "video":
            feat_ext = get_video_feature_extractor(layer=layer, mod_type=feat_ext_type, device=device, use_pretrained=use_pretrained)
            feat_ext = VideoFeatureExtractor(feat_ext, stim_shape, device=device)
        elif modality == "image":
            feat_ext = get_image_feature_extractor(layer=layer, mod_type=feat_ext_type, device=device, use_pretrained=use_pretrained)
            feat_ext = ImageFeatureExtractor(feat_ext, stim_shape, device=device)

        readout_input = feat_ext(train_dataset[0][0].unsqueeze(0).to(device))
        print(f"readout input shape: {readout_input.shape}")
        if modality == "video" and flatten_time:
            num_input  = readout_input.shape[1] * readout_input.shape[2]
        else:
            num_input = readout_input.shape[1]
            
        num_neurons = len(train_dataset[0][1])

        for param in feat_ext.parameters():
            param.requires_grad = False

        if feat_ext_type == 'hiera':
            readout = TransformerReadout(num_input=num_input, num_neurons=num_neurons, device=device)
        else:
            readout = PoissonGaussianReadout(num_input, num_neurons, use_sigma = use_sigma, use_pool = use_pool, pool_size = pool_size, pool_stride= pool_stride, center_readout = center_readout, modality=modality, flatten_time=flatten_time, mlp=mlp, device=device)

        self.model = nn.Sequential(
            feat_ext,
            readout
        )
    def forward(self, x):
        return self.model(x)