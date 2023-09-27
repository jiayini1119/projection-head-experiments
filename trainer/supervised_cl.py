import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from evaluator import Evaluator_PH
from spuco.robust_train import BaseRobustTrain
from spuco.utils.random_seed import seed_randomness
from spuco.utils import Trainer
from loss.supcon_loss import SupConLoss


def forward_pass(self, batch):  
    inputs, labels = batch
    inputs, labels = inputs.to(self.device), labels.to(self.device)
    outputs = self.model.get_representation(inputs, use_ph=True)
    loss = self.criterion(outputs, labels)
    return loss, outputs, labels

def forward_pass_ori(self, batch):  
    inputs, labels = batch
    inputs, labels = inputs.to(self.device), labels.to(self.device)
    outputs = self.model.get_representation(inputs, use_ph=False)
    loss = self.criterion(outputs, labels)
    return loss, outputs, labels

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
        verbose=False,
        use_ph=True,
    ):
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        super().__init__(val_evaluator=val_evaluator, verbose=verbose)

        self.num_epochs = num_epochs

        if use_ph:
            self.trainer = Trainer(
                trainset=trainset,
                model=model,
                batch_size=batch_size,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                max_grad_norm=max_grad_norm,
                criterion=criterion,
                forward_pass=forward_pass,
                verbose=verbose,
                device=device
            )
        else:
            self.trainer = Trainer(
                trainset=trainset,
                model=model,
                batch_size=batch_size,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                max_grad_norm=max_grad_norm,
                criterion=criterion,
                forward_pass=forward_pass_ori,
                verbose=verbose,
                device=device
            )





