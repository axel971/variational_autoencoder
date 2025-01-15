import torch.nn as nn
import torch
import torchmetrics
from tqdm import tqdm
from models.testing_functions import eval_step

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               metric_fn: torchmetrics,
               device: torch.device
               ):

    loss_value = 0

    loop = tqdm(dataloader)

    model.train() # Put the model in training mode (put in training mode batchnorm, dropout, etc)

    for(x, _) in loop:

        x = x.to(device) # To do: reshape, view the image
        
        x_flattened = x.flatten(start_dim = 1, end_dim = -1)
        x_flattened_pred, mean, logVar = model(x_flattened)

        # Compute loss of the current batch
        loss = loss_fn(x_flattened_pred, x_flattened, mean, logVar)
        loss_value += loss.item()

        # Compute metric of the current batch
        metric = metric_fn(x_flattened_pred, x_flattened)

        # Display loss and metric for the current batch
        loop.set_postfix(loss = loss.item(), metric = metric.item())

        # Update the model weights
        optimizer.zero_grad()

        # Perform back propagation
        loss.backward()
        # Update the new weights
        optimizer.step()

    # Loss and metric value after one epoch
    loss_value = loss_value/len(dataloader)
    metric_value = metric_fn.compute()

    metric_fn.reset()

    return loss_value, metric_value

def train(model: nn.Module,
          training_dataloader: torch.utils.data.DataLoader,
          testing_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          metric_fn: torchmetrics,
          epochs: int,
          device: torch.device):


    results = {"Train_loss": [],
               "Train_metric": [],
               "Test_loss": [],
               "Test_metric": []
               }

    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}/{epochs}:")

        train_loss_value, train_metric_value = train_step(model = model,
                                                          dataloader = training_dataloader,
                                                          loss_fn = loss_fn,
                                                          optimizer = optimizer,
                                                          metric_fn = metric_fn,
                                                          device = device)

        test_loss_value, test_metric_value = eval_step(model = model,
                                                             dataloader = testing_dataloader,
                                                             loss_fn = loss_fn,
                                                             metric_fn = metric_fn,
                                                             device = device)

        print(f"Epoch {epoch + 1} | Train loss: {train_loss_value} | Train metric: {train_metric_value} | Test loss: {test_loss_value} | Test metric: {test_metric_value}")
        

        results["Train_loss"].append(train_loss_value)
        results["Train_metric"].append(train_metric_value)
        results["Test_loss"].append(test_loss_value)
        results["Test_metric"].append(test_metric_value)

    return results
