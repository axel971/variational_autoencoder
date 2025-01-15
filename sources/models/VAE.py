import torch
from torch import nn

# Input img -> Hiden dim -> mean, std -> Parametrization trick -> Decoder -> Output img
class VAEModel(nn.Module):
        
    def __init__(self, input_dim, hidden_dim_1 = 400, hidden_dim_2 = 200, z_dim = 20):
        super().__init__()

        # Encoder
        self.encoder_layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim_1),
                nn.Mish(),
                nn.Linear(hidden_dim_1, hidden_dim_2),
                nn.Mish()        
                )

        self.mean_layer = nn.Linear(hidden_dim_2, z_dim)
        self.logVar_layer = nn.Linear(hidden_dim_2, z_dim)

        # Decoder
        self.decoder_layers = nn.Sequential(
                nn.Linear(z_dim, hidden_dim_2),
                nn.Mish(),
                nn.Linear(hidden_dim_2, hidden_dim_1),
                nn.Mish(),
                nn.Linear(hidden_dim_1, input_dim),
                nn.Sigmoid()
                )

    def encoder(self, x):
        # q_phi(z|x)

        h = self.encoder_layers(x)
        mean, logVar = self.mean_layer(h), self.logVar_layer(h)

        return mean, logVar

    def decoder(self, z):
        # p_theta(x|z)

        return self.decoder_layers(z)
        

    def sampling_with_reparametrizationTrick(self, mean, logVar):

        std = torch.exp(0.5*logVar)

        epsilon = torch.randn_like(std) # To Do: Compute the epsilone with a standard gaussian distribution

        return mean + std*epsilon


    def forward(self, x):

        mean, logVar = self.encoder(x)

        z_reparametrized = self.sampling_with_reparametrizationTrick(mean, logVar)

        x_reconstructed = self.decoder(z_reparametrized)

        return x_reconstructed, mean, logVar
