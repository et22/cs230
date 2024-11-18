import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

## transformer readout
class TransformerReadout(nn.Module):
    def __init__(self, num_input=768, num_neurons=49, modality="image", device=torch.device("cpu")):
        super().__init__()

        self.modality = modality 
        self.device = device

        # gaussian 
        self.num_neurons = num_neurons
        
        # linear
        self.linear_layers = nn.ModuleList([nn.Linear(num_input, 1, device=device) for _ in range(num_neurons)])
        
        self.linear = nn.Linear(num_input, num_neurons, device=device)
        self.act = nn.ELU()

    def forward(self, x):        
        x = torch.diagonal(self.linear(x.permute(0, 2, 1)), offset=0, dim1=1, dim2=2)
        x = self.act(x) + 1
        return x

## pytorch poisson readout
class PoissonGaussianReadout(nn.Module):
    def __init__(self, num_input, num_neurons, modality="image", use_pool = False, pool_size = 2, pool_stride = 2, use_sigma = True, center_readout=False, flatten_time=False, mlp=False, device=torch.device("cpu")):
        super().__init__()

        self.modality = modality 
        self.device = device

        self.center_readout = center_readout
        self.use_sigma = use_sigma
        self.use_pool = use_pool

        self.flatten_time = flatten_time
        
        # gaussian 
        self.num_neurons = num_neurons
        self.mu = nn.Parameter(torch.zeros((num_neurons, 2), requires_grad=True, device=device))
        # was 0.3!
        self.sigma = nn.Parameter(0.3 * torch.ones((num_neurons), requires_grad=True, device=device))
        self.normal = MultivariateNormal(loc=torch.zeros(2).to(self.device), covariance_matrix=torch.eye(2).to(self.device))

        # linear
        self.linear_layers = nn.ModuleList([nn.Linear(num_input, 1, device=device) for _ in range(num_neurons)])
        self.linear = nn.Linear(num_input, num_neurons, device=device)
        self.act = nn.ELU()

        # pooling size
        self.pool = nn.AvgPool2d(pool_size, stride=pool_stride, padding=int(pool_size/2), count_include_pad=False)

        self.mlp = mlp
        if self.mlp:
            self.act1 = nn.ReLU()
            self.linear1 = nn.Linear(num_input, num_input, device=device)
        
    def forward(self, x):
        if self.modality == "video":
            if self.flatten_time:
                n_batch, n_channel, n_time, width, height = x.shape
                x = x.view(n_batch, n_channel * n_time, width, height)
            else:
                x = x[:, :, -1, :, :] # get embedding of last "frame"
            
        if self.use_pool:
            x = self.pool(x)
            
        if self.training and self.use_sigma:
            noise = self.normal.rsample((x.shape[0], 1, 1))
            grid = self.mu.view(1, self.num_neurons, 1, 2) + self.sigma.view(1, self.num_neurons, 1, 1) * noise 
        else:
            grid = self.mu.view(1, self.num_neurons, 1, 2) * torch.ones((x.shape[0], self.num_neurons, 1, 2), device=self.device)

        if self.center_readout:
            grid = torch.zeros((x.shape[0], self.num_neurons, 1, 2), device=self.device)

        grid = torch.clamp(grid, min=-1, max=1) # clamp to ensure within feature map

        x = torch.squeeze(torch.squeeze(F.grid_sample(x, grid, align_corners=False), -1), -1)
        if self.mlp:
            x = self.act1(self.linear1(x.permute(0, 2, 1))).permute(0, 2, 1)
            
        x = torch.diagonal(self.linear(x.permute(0, 2, 1)), offset=0, dim1=1, dim2=2)
        x = self.act(x) + 1
        return x