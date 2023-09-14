import random
import torch
import numpy as np
from spuco.evaluate import Evaluator
from torch.utils.data import DataLoader

from spuco.utils.random_seed import seed_randomness

class Evaluator_PH(Evaluator):
    def __init__(
        self,
        use_ph: bool = False,
        *args,
        **kwargs
    ):
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        super().__init__(*args, **kwargs)
        self.use_ph = use_ph
    
    def _encode_testset(self, testloader):
        X_test = []
        y_test = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                X_test.append(self.model.get_representation(inputs, use_ph=self.use_ph))
                y_test.append(labels)
            return torch.cat(X_test), torch.cat(y_test)

    def _evaluate_accuracy(self, testloader: DataLoader):
        with torch.no_grad():
            correct = 0
            total = 0    
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs, use_ph=self.use_ph)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            return 100 * correct / total