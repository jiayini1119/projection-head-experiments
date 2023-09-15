"""
Adapted from https://github.com/sjoshi804/sas-data-efficient-contrastive-learning/blob/311a976f47da034fcb59c185a7c7a8d4c5f936dc/projection_heads/critic.py#L5
"""

import torch
from torch import nn

class LinearCritic(nn.Module):

    def __init__(self, temperature=0.5):
        super(LinearCritic, self).__init__()
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)

    def compute_sim(self, z):
        p = []
        for i in range(len(z)):
            p.append(z[i])
        
        sim = {}
        for i in range(len(p)):
            for j in range(i, len(p)):
                sim[(i, j)] = self.cossim(p[i].unsqueeze(-2), p[j].unsqueeze(-3)) / self.temperature
        

        d = sim[(0,0)].shape[-1]
        if (0, 1) in sim:
            assert sim[(0,0)].shape[-1] == sim[(0,1)].shape[-1]
        # d = sim[(0,1)].shape[-1]

        for i in range(len(p)):
            sim[(i,i)][..., range(d), range(d)] = float('-inf')  
        

        for i in range(len(p)):
            sim[i] = torch.cat([sim[(j, i)].transpose(-1, -2) for j in range(0, i)] + [sim[(i, j)] for j in range(i, len(p))], dim=-1)
        sim = torch.cat([sim[i] for i in range(len(p))], dim=-2)
        
        return sim

    def forward(self, z):
        return self.compute_sim(z)