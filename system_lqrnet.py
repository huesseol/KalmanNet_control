import torch
from torch import nn
import torch.nn.functional as func

class LQRNet(nn.Module):
    def __init__(self):
        super().__init__()


    def InitLQRNet(self, state_dim, input_dim, L):
        # Linear map from state to input: u = -Lx
        # Learn parameters for L
        self.linear = nn.Linear(state_dim, input_dim, bias=False)
        # Initialize weights with an estimate of L 
        # (obtained from a possibly wrong model)
        self.linear.weight.data = L


    def forward(self, x):
        # Normalize x
        norm_x = torch.linalg.norm(x)
        x_norm = x / norm_x
        
        u_norm = self.linear(x_norm)
        # Denormalize
        u = u_norm * norm_x
        return u



class LQRNet2(nn.Module):
    def __init__(self):
        super().__init__()

    
    def InitLQRNet(self, input_size, output_size, layer_sizes=()):
        sizes = (input_size + 1,) + layer_sizes + (output_size,)
        num_affine_maps = len(sizes) - 1
        self.layers = nn.ModuleList([
            nn.Linear(sizes[k], sizes[k+1], bias=True) for k in range(num_affine_maps)
        ])
        self.activation = nn.ReLU()


    def forward(self,x):
        # Normalize x
        norm_x = torch.linalg.norm(x)
        x_norm = x / norm_x
        # Include norm as input feature
        layer_input = torch.concat((x_norm, norm_x.unsqueeze(0)))
        for idx, layer in enumerate(self.layers):
            layer_output = layer(layer_input)
            if idx < len(self.layers) - 1:
                layer_output = self.activation(layer_output)
            layer_input = layer_output
        net_output = layer_output
        return net_output

    
class LQRNet3(nn.Module):
    def __init__(self):
        super().__init__()

    
    def InitLQRNet(self, input_size, output_size, layer_sizes=()):
        sizes = (input_size,) + layer_sizes + (output_size,)
        num_affine_maps = len(sizes) - 1
        self.layers = nn.ModuleList([
            nn.Linear(sizes[k], sizes[k+1], bias=True) for k in range(num_affine_maps)
        ])
        self.activation = nn.ReLU()


    def forward(self,x):
        # Normalize x
        norm_x = torch.linalg.norm(x)
        x_norm = x / norm_x
        # Include norm as input feature
        layer_input = x_norm
        for idx, layer in enumerate(self.layers):
            layer_output = layer(layer_input)
            if idx < len(self.layers) - 1:
                layer_output = self.activation(layer_output)
            layer_input = layer_output
        net_output = layer_output * norm_x
        return net_output

    