import random

import numpy as np
import torch
from torch import nn

from typing import Optional
from spuco.utils.random_seed import seed_randomness

import torch.nn.functional as F

class ReweightProjectionHead(nn.Module):
    def __init__(self, representation_dim, kappa):
        super(ReweightProjectionHead, self).__init__()
        self.representation_dim = representation_dim
        self.kappa = kappa
        if self.kappa is not None:
            self.reweights = nn.Parameter(torch.tensor([1/self.kappa**i for i in range(representation_dim)]))
            self.reweights.requires_grad = False
        else:
            self.reweights = None

    def forward(self, x):
        if self.reweights is not None:
            x = x * self.reweights
        else:
            raise ValueError("no projection head used")
        return x


class SymmetricReLU(nn.Module):
    def __init__(self, b=0.0):
        super(SymmetricReLU, self).__init__()
        self.b = nn.Parameter(torch.tensor(b))  

    def forward(self, x):
        relu_pos = F.relu(x - self.b)
        relu_neg = F.relu(-x - self.b)
        return relu_pos - relu_neg

class SpuCoModel(nn.Module):
    """
    Wrapper module to allow for methods that use penultimate layer embeddings
    to easily access this via *backbone*
    """
    def __init__(
        self,
        backbone: nn.Module, 
        representation_dim: int,
        num_classes: int,
        kappa: Optional[float] = None,
        mult_layer: bool = False,
        identity_init: bool = False,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initializes a SpuCoModel 

        :param backbone: The backbone network.
        :type backbone: torch.nn.Module
        :param representation_dim: The dimensionality of the penultimate layer embeddings.
        :type representation_dim: int
        :param num_classes: The number of output classes.
        :type num_classes: int
        """
        
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
        super().__init__()
        # Encoder
        self.backbone = backbone 
        self.representation_dim = representation_dim

        self.kappa = kappa

        # Projection head 
        if hidden_dim is not None:
            if not mult_layer:
                if not identity_init:
                    # self.projection_head = nn.Sequential(
                    #     nn.Linear(representation_dim, hidden_dim),
                    #     nn.ReLU(),
                    #     nn.Linear(hidden_dim, representation_dim),
                    # )

                    # change
                    self.projection_head = ReweightProjectionHead(representation_dim=representation_dim, kappa=kappa)

                    for name, param in self.projection_head.named_parameters():
                        print(name, param)


                else:
                    # initialize the projection head as an identity function
                    self.projection_head = nn.Sequential(
                        nn.Linear(representation_dim, hidden_dim),
                        SymmetricReLU(),
                        nn.Linear(hidden_dim, representation_dim),
                    )
                     
                    self.projection_head[0].weight.data.copy_(torch.eye(representation_dim))
                    self.projection_head[0].bias.data.fill_(0)
                    self.projection_head[2].weight.data.copy_(torch.eye(hidden_dim))
                    self.projection_head[2].bias.data.fill_(0)
            else:
                self.projection_head = nn.Sequential(
                    nn.Linear(representation_dim, hidden_dim), 
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),  
                    nn.ReLU(),
                    nn.Linear(hidden_dim, representation_dim)  
                )
        else:
            self.projection_head = None

        self.classifier = nn.Linear(representation_dim, num_classes)
    
    def get_representation(self, x, use_ph: bool = False):
        """
        Get representation out of the encoder or the projection head

        :param x: Input tensor.
        :type x: torch.Tensor
        :param use_ph: Whether to use projection head
        :type use_ph: bool
        :return: Output embedding tensor.
        :rtype: torch.Tensor
        """

        if use_ph and self.projection_head is None:
                    raise ValueError("No projection head for the model but use_ph is True")

        encoder_rep = self.backbone(x)
        return self.projection_head(encoder_rep) if use_ph else encoder_rep
        # return encoder_rep * self.reweight if use_ph else encoder_rep


    def forward(self, x, use_ph: Optional[bool] = None):
        """
        Forward pass of the SpuCoModel.

        :param x: Input tensor.
        :type x: torch.Tensor
        :param use_ph: Whether to use projection head
        :type use_ph: bool
        :return: Output tensor.
        :rtype: torch.Tensor
        """

        if use_ph and self.projection_head is None:
            raise ValueError("No projection head for the model but use_ph is True")
        
        if use_ph is None:
            use_ph = self.projection_head is not None
        encoder_rep = self.backbone(x)
        return self.classifier(self.projection_head(encoder_rep)) if use_ph else self.classifier(encoder_rep)

        # return self.classifier(self.reweight * encoder_rep) if use_ph else self.classifier(encoder_rep)