import random
import torch
import numpy as np
from spuco.last_layer_retrain import DFR
from torch.utils.data import DataLoader
from spuco.utils.random_seed import seed_randomness
from spuco.datasets import GroupLabeledDatasetWrapper
from models.spuco_model import SpuCoModel
from sklearn.preprocessing import StandardScaler
from torch import nn


class DFR_PH(DFR):
    def __init__(
        self,
        model: SpuCoModel,
        use_ph: bool = False,
        *args,
        **kwargs
    ):
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        super().__init__(*args, model=model, **kwargs)
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
            num_workers=5, 
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



class DFR_MLP(DFR_PH):
    def __init__(
        self,
        num_epochs: int = 1000,
        weight_decay: float = 0,
        lr: float = 1e-1,
        hidden_dim: int = 2048,
        *args,
        **kwargs
    ):
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        super().__init__(*args, **kwargs)

        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.lr = lr
        self.hidden_dim = hidden_dim


    def train_single_model(self, X_train, y_train, g_train):
        group_names = {g for g in g_train}
        group_partition = []
        for g in group_names:
            group_partition.append(np.where(g_train==g)[0])
        min_size = np.min([len(g) for g in group_partition])
        X_train_balanced = []
        y_train_balanced = []
        for g in group_partition:
            indices = np.random.choice(g, size=min_size, replace=False)
            X_train_balanced.append(X_train[indices])
            y_train_balanced.append(y_train[indices])
        X_train_balanced = np.concatenate(X_train_balanced)
        y_train_balanced = np.concatenate(y_train_balanced)

        X_train_balanced = torch.from_numpy(X_train_balanced).float().to(self.device)
        y_train_balanced = torch.from_numpy(y_train_balanced).long().to(self.device)


        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(X_train_balanced.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )

        self.mlp.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.mlp.parameters(), weight_decay=self.weight_decay, lr=self.lr)
    
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.num_epochs)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            pred = self.mlp(X_train_balanced)
            loss = criterion(pred, y_train_balanced)
            loss.backward()
            optimizer.step()
            schedule.step()
            acc = (torch.argmax(pred, -1) == y_train_balanced).detach().float().mean()
            if self.verbose and epoch % (self.num_epochs // 10) == 0:
                print(epoch, acc)

    def train(self):
        """
        Retrain MLP
        """
        if self.verbose:
            print('Encoding data ...')
        X_labeled, y_labeled, g_labeled = self.encode_dataset(self.group_labeled_set)
        X_labeled = X_labeled.detach().cpu().numpy()
        y_labeled = y_labeled.detach().cpu().numpy()
        g_labeled = g_labeled.detach().cpu().numpy()

        # Standardize features
        if self.preprocess:
            self.scaler = StandardScaler()
            if self.data_for_scaler:
                X_scaler, _ = self.encode_dataset(self.data_for_scaler)
                X_scaler = X_scaler.detach().cpu().numpy()
                self.scaler.fit(X_scaler)
            else:
                self.scaler = StandardScaler()
                self.scaler.fit(X_labeled)
            X_labeled = self.scaler.transform(X_labeled)
    
        self.train_single_model(X_labeled, y_labeled, g_labeled)