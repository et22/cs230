import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from fix_models.feature_extractors import get_video_feature_extractor, VideoFeatureExtractor

from fix_models.readouts import PoissonGaussianReadout, PoissonLinearReadout
        
class FullModel(nn.Module):
    def __init__(self, modality, layer, stim_shape, train_dataset, feat_ext_type = 'none', use_pool = False, pool_size = 2, pool_stride = 2, use_pretrained = True, freeze_weights=True, flatten_time = False, device=torch.device("cpu")):
        super().__init__()
        num_neurons = len(train_dataset[0][1])

        if feat_ext_type != 'none':
            feat_ext = get_video_feature_extractor(layer=layer, mod_type=feat_ext_type, device=device, use_pretrained=use_pretrained, freeze_weights=freeze_weights)
            feat_ext = VideoFeatureExtractor(feat_ext, stim_shape, device=device)
            
            readout_input = feat_ext(train_dataset[0][0].unsqueeze(0).to(device))
            num_input  = readout_input.shape[1] * readout_input.shape[2]
            
            readout = PoissonGaussianReadout(num_input, num_neurons, use_pool = use_pool, pool_size = pool_size, pool_stride= pool_stride, modality=modality, device=device)
            
            self.model = nn.Sequential(
                feat_ext,
                readout
            )
        else:
            num_input = train_dataset[0][0].flatten().shape[0]
            readout = PoissonLinearReadout(num_input, num_neurons, modality=modality, device=device)
            self.model = nn.Sequential(
                nn.Flatten(),
                readout
            )
            
        print(f"readout input shape: {num_input}")

    def forward(self, x):
        return self.model(x)