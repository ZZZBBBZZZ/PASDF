import torch.nn as nn
import torch
import copy
from tqdm import tqdm
from utils import utils_deepsdf
import numpy as np
"""
The term 'ofo' means 'one for one'.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SDFModel_ofo_PoseEmbedder(torch.nn.Module):
    def __init__(self, num_layers, skip_connections, PoseEmbedder_size, inner_dim=512, output_dim=1):
        """
        SDF model for multiple shapes.
        Args:
            input_dim: 60 for PoseEmbedder + 3 points = 63
        """
        super(SDFModel_ofo_PoseEmbedder, self).__init__()

        # Num layers of the entire network
        self.num_layers = num_layers 

        # If skip connections, add the input to one of the inner layers
        self.skip_connections = skip_connections

        self.PoseEmbedder_size = PoseEmbedder_size   

        # Dimension of the input space (3D coordinates)
        dim_coords = 3 
        input_dim = self.PoseEmbedder_size + dim_coords

        # Copy input size to calculate the skip tensor size
        self.skip_tensor_dim = copy.copy(input_dim)

        # Compute how many layers are not Sequential
        num_extra_layers = 2 if (self.skip_connections and self.num_layers >= 8) else 1
        
        # Add sequential layers
        layers = []
        for _ in range(num_layers - num_extra_layers):
            layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), nn.ReLU()))
            input_dim = inner_dim
        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(nn.Linear(inner_dim, output_dim), nn.Tanh())
        self.skip_layer = nn.Sequential(nn.Linear(inner_dim, inner_dim - self.skip_tensor_dim), nn.ReLU())


    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, 131). It contains a stacked tensor [latent_code, samples].
        Returns:
            sdf: output tensor of shape (batch_size, 1)
        """      
        input_data = x.clone().detach()

        # Forward pass
        if self.skip_connections and self.num_layers >= 5:
            for i in range(3):
                x = self.net[i](x)
            x = self.skip_layer(x)
            x = torch.hstack((x, input_data))
            for i in range(self.num_layers - 5):
                x = self.net[3 + i](x)
            sdf = self.final_layer(x)
        else:
            if self.skip_connections:
                print('The network requires at least 5 layers to skip connections. Normal forward pass is used.')
            x = self.net(x)
            sdf = self.final_layer(x)
        return sdf


