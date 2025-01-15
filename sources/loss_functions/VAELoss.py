
import torch
import torch.nn as nn


class VAELoss(nn.Module):
    
    def __init__(self):

        super().__init__()

    def forward(self, inputs, targets, mean, logVar):

        reconstruction_loss = nn.functional.mse_loss(inputs, targets, reduction ="sum")

        kl_divergence = -0.5*torch.sum(1 + logVar - mean.pow(2) - logVar.exp())

        return reconstruction_loss + kl_divergence

