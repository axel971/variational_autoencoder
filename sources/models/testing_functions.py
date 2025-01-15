import torch.nn as nn
import torch
import torchmetrics
from typing import Tuple

def eval_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              metric_fn: torchmetrics,
              device: torch.device):

    loss_value = 0

    model.eval()
    with torch.inference_mode():

        for (x, _) in dataloader:
            
            x = x.to(device)

            x_flattened = x.flatten(start_dim = 1, end_dim = -1)

            x_flattened_pred, mean, logVar = model(x_flattened)

            loss = loss_fn(x_flattened_pred, x_flattened, mean, logVar)
            loss_value += loss.item()

            metric_fn(x_flattened_pred, x_flattened)

    # Compute the loss and metric for the current epoch

    loss_value /= len(dataloader)
    metric_value = metric_fn.compute()

    metric_fn.reset()

    return loss_value, metric_value

def predictSample(model: nn.Module,
                  z: torch.Tensor,
                  image_size: Tuple,
                  device: torch.device):

    model.eval()
    with torch.inference_mode():
        z = z.to(device).unsqueeze(dim = 0)
        output_reconstructed_image_flattened = model.decoder(z)
    
    return output_reconstructed_image_flattened.unflatten(dim = 1, sizes = image_size).squeeze(dim = 0)




