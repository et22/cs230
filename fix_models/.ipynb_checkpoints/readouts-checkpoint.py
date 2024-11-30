import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

## poisson linear readout
class PoissonLinearReadout(nn.Module):
    def __init__(self, num_input, num_neurons, modality="video", device=torch.device("cpu")):
        super().__init__()

        self.modality = modality 
        self.device = device

        # gaussian 
        self.num_neurons = num_neurons
        
        # linear        
        self.linear = nn.Linear(num_input, num_neurons, device=device)
        self.act = nn.ELU()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1)
            x = torch.diagonal(self.linear(x), offset=0, dim1=1, dim2=2)
        else:
            x = self.linear(x)
        x = self.act(x) + 1
        return x

## poisson gaussian readout
class PoissonGaussianReadout(nn.Module):
    def __init__(self, num_input, num_neurons, modality="video", use_pool = False, pool_size = 2, pool_stride = 2, device=torch.device("cpu")):
        super().__init__()

        self.modality = modality 
        self.device = device

        self.use_pool = use_pool
        
        # gaussian 
        self.num_neurons = num_neurons
        self.mu = nn.Parameter(torch.zeros((num_neurons, 2), requires_grad=True, device=device))
        # was 0.3!
        self.sigma = nn.Parameter(0.3 * torch.ones((num_neurons), requires_grad=True, device=device))
        self.normal = MultivariateNormal(loc=torch.zeros(2).to(self.device), covariance_matrix=torch.eye(2).to(self.device))

        # linear
        self.poisson_linear = PoissonLinearReadout(num_input, num_neurons, modality, device)

        # pooling size
        self.pool = nn.AvgPool2d(pool_size, stride=pool_stride, padding=int(pool_size/2), count_include_pad=False)
        
    def forward(self, x):
        n_batch, n_channel, n_time, width, height = x.shape
        x = x.view(n_batch, n_channel * n_time, width, height)
        
        if self.use_pool:
            x = self.pool(x)
            
        if self.training:
            noise = self.normal.rsample((x.shape[0], 1, 1))
            grid = self.mu.view(1, self.num_neurons, 1, 2) + self.sigma.view(1, self.num_neurons, 1, 1) * noise 
        else:
            grid = self.mu.view(1, self.num_neurons, 1, 2) * torch.ones((x.shape[0], self.num_neurons, 1, 2), device=self.device)

        grid = torch.clamp(grid, min=-1, max=1) # clamp to ensure within feature map
        
        x = torch.squeeze(torch.squeeze(F.grid_sample(x, grid, align_corners=False), -1), -1)

        x = self.poisson_linear(x)
        
        return x