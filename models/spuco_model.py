import random

import numpy as np
import torch
from torch import nn

from typing import Optional
from spuco.utils.random_seed import seed_randomness

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

        # Projection head 
        if hidden_dim is not None:
            self.projection_head = nn.Sequential(
                nn.Linear(representation_dim, hidden_dim),
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

        # print("get representation", use_ph)

        encoder_rep = self.backbone(x)
        return self.projection_head(encoder_rep) if use_ph else encoder_rep


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

        # print("forward", use_ph)

        encoder_rep = self.backbone(x)
        return self.classifier(self.projection_head(encoder_rep)) if use_ph else self.classifier(encoder_rep)