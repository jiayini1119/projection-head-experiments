import random
import torch
import numpy as np
from spuco.last_layer_retrain import DFR
from torch.utils.data import DataLoader
from spuco.utils.random_seed import seed_randomness
from spuco.datasets import GroupLabeledDatasetWrapper


class DFR_PH(DFR):
    def __init__(
        self,
        use_ph: bool = False,
        *args,
        **kwargs
    ):
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        super().__init__(*args, **kwargs)
        self.use_ph = use_ph

    def encode_dataset(self, dataset):
        labeled = type(dataset) == GroupLabeledDatasetWrapper

        X_train = []
        y_train = []
        if labeled:
            g_train = []

        trainloader = DataLoader(
            dataset=dataset, 
            batch_size=100,
            shuffle=False,
            num_workers=4, 
            pin_memory=True
        )

        self.model.eval()
        with torch.no_grad():
            for batch in trainloader:
                if labeled:
                    inputs, labels, groups = batch
                    inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)
                    g_train.append(groups)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                X_train.append(self.model.get_representation(inputs, use_ph=self.use_ph))

                y_train.append(labels)
                    
            if labeled:
                return torch.cat(X_train), torch.cat(y_train), torch.cat(g_train)
            else:
                return torch.cat(X_train), torch.cat(y_train)