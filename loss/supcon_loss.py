import torch
import torch.nn as nn
from loss.critic import LinearCritic


class SupConLoss(nn.Module):
   
    def __init__(self, temperature: float=0.5, num_positive: int=1):
        super(SupConLoss, self).__init__()
        self.critic = LinearCritic(temperature=temperature)
        self.num_positive = num_positive
       

    def forward(self, z, labels):
        # Adapted from https://github.com/sjoshi804/sas-data-efficient-contrastive-learning/blob/311a976f47da034fcb59c185a7c7a8d4c5f936dc/trainer.py#L58
        batch_size = int(len(z) / self.num_positive)
        aug_z = []
        for i in range(self.num_positive):
            aug_z.append(z[i * batch_size : (i + 1) * batch_size])
        z = aug_z

        sim = self.critic(z)
        log_sum_exp_sim = torch.log(torch.sum(torch.exp(sim), dim=1))
        targets = torch.cat([labels] * self.num_positive)
        pos_pairs = (targets.unsqueeze(1) == targets.unsqueeze(0))
        inf_mask = (sim != float('-inf'))
        pos_pairs = torch.logical_and(pos_pairs, inf_mask)

        pos_count = torch.sum(pos_pairs, dim=1)
        pos_sims = torch.nansum(sim * pos_pairs, dim=-1)
        res = torch.mean(-pos_sims / pos_count + log_sum_exp_sim)
        if torch.isnan(res):
            raise ValueError("supcon_loss is nan!")
        return torch.mean(-pos_sims / pos_count + log_sum_exp_sim)
        
        