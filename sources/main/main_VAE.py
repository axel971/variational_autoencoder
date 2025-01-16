
# data_dir = C:\Users\axell\Documents\dev\variational_autoencoder\data"
# python main_VAE.py -data_dir "C:\Users\axell\Documents\dev\variational_autoencoder\data"

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchmetrics.regression import MeanAbsoluteError
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import random

import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from models.VAE import VAEModel
from models.training_functions import train
from models.testing_functions import predictSample
from loss_functions.VAELoss import VAELoss

def main(data_dir: str):

    DATA_DIR = Path(data_dir)
    DATA_DIR.mkdir(parents = True, exist_ok = True)
    
    # Create training and testing dataset
    training_dataset = datasets.MNIST(root = DATA_DIR,
                                      train = True,
                                      download = True,
                                      transform = ToTensor()
                                      )

    testing_dataset = datasets.MNIST(root = DATA_DIR,
                                     train = False,
                                     download = True,
                                     transform = ToTensor()
                                     )

    class_names = training_dataset.classes
    num_classes = int(len(class_names))
    image, label = training_dataset[0]
    img_size = image.size()



    # Create dataloader
    BATCH_SIZE = 128
    training_dataloader = DataLoader(dataset = training_dataset,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True)

    testing_dataloader = DataLoader(dataset = testing_dataset,
                                    batch_size = BATCH_SIZE,
                                    shuffle = False)

    # Setup device agnostic code
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = VAEModel(input_dim = img_size[0]*img_size[1]*img_size[2]).to(device)
    
    # Instantiate the loss, optimizer, and metric
    loss_fn = VAELoss()
    optimizer = Adam(params = model.parameters(),
                     lr = 1e-3)
    metric_fn = MeanAbsoluteError()

    # Train the model
    EPOCHS = 100

    train(model = model,
          training_dataloader = training_dataloader,
          testing_dataloader = testing_dataloader,
          loss_fn = loss_fn,
          optimizer = optimizer,
          metric_fn = metric_fn,
          epochs = EPOCHS,
          device = device)

    # Display random examples
    nExamples = 5
    random_example_idxs = random.sample(range(len(testing_dataset)), k = nExamples)
    
    plt.figure(figsize=(12, 9))

    for i, image_idx in enumerate(random_example_idxs):
        img, label = testing_dataset[image_idx]
    
        model.eval()
        with torch.inference_mode():
            reconstructed_img_flattened, _, _ = model(img.unsqueeze(dim = 0).flatten(start_dim = 1, end_dim = - 1))
        reconstructed_img = reconstructed_img_flattened.unflatten(dim = 1, sizes = img_size).squeeze(dim = 0)

        plt.subplot(nExamples, 2, (i*2) +1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.axis(False)
        if i == 0:
            plt.title("Original image")
        

        plt.subplot(nExamples, 2, (i*2) +2)
        plt.imshow(reconstructed_img.squeeze().detach().numpy(), cmap="gray")
        plt.axis(False)
        if i == 0:
            plt.title("Reconstructed image")
        
        
    plt.savefig("OriginalImages_and_ReconstructedImages.png")
    plt.show()


    # Generate and display an artificial image with model
    z_dim = 20
    z = torch.rand(z_dim)

    image_generated = predictSample(model = model,
                                    z = z,
                                    image_size = img_size,
                                    device = device)

    plt.figure(figsize=(9,7))
    plt.imshow(image_generated.squeeze().detach().numpy(), cmap = "gray")
    plt.axis(False)
    plt.title("Generated image")
    plt.show()

    return


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", required = True, help = "Path toward data directory")
    args = parser.parse_args()
    main(data_dir = args.data_dir)


