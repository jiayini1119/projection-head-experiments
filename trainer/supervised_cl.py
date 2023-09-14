import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from evaluator import Evaluator_PH
from spuco.robust_train import BaseRobustTrain
from spuco.utils.random_seed import seed_randomness
from spuco.utils import Trainer
from util.supcon_loss import SupConLoss

class SCL(BaseRobustTrain):
    """
    Supervised Contrastive Learning
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: Dataset,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        criterion: SupConLoss,
        device: torch.device = torch.device("cpu"),
        lr_scheduler=None,
        max_grad_norm=None,
        val_evaluator: Evaluator_PH = None,
        verbose=False
    ):
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        super().__init__(val_evaluator=val_evaluator, verbose=verbose)

        self.num_epochs = num_epochs

        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            criterion=criterion,
            verbose=verbose,
            device=device
        )




