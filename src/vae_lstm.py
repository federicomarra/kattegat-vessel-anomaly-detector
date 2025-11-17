import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int=128, num_layers: int=1, z_dim: int=32, dropout: float=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.hidden_to_mu = nn.Linear(hidden_size, z_dim)
        self.hidden_to_logvar = nn.Linear(hidden_size, z_dim)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        # x: (B, T, in_dim), mask: (B, T)
        out, _ = self.lstm(x)  # out: (B, T, H)
        # weighted average over time steps using the mask
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, H)
        mu = self.hidden_to_mu(pooled)          # (B, z_dim)
        logvar = self.hidden_to_logvar(pooled)  # (B, z_dim)
        return mu, logvar
    
class DecoderLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int=128, num_layers: int=1, z_dim: int=5, dropout: float=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.lstm = nn.LSTM(input_size=in_dim + z_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.z_to_hidden = nn.Linear(z_dim, hidden_size)
        self.hidden_to_out = nn.Linear(hidden_size, in_dim)

    def forward(self, x:torch.Tensor, z:torch.Tensor):
        # x: (B, T, in_dim), z: (B, Z)
        B, T, F = x.shape
        start = torch.zeros((B, 1, F), device=x.device, dtype=x.dtype)  # (B, 1, F)
        x_shifted = torch.cat([start, x[:, :-1, :]], dim=1)  # (B, T, F)

        z_rep = z.unsqueeze(1).expand(B, T, z.shape[-1])  # (B, T, Z)
        dec_in = torch.cat([x_shifted, z_rep], dim=-1)  # (B, T, F + Z)

        # initial state 
        h0 = torch.tanh(self.z_to_hidden(z)).unsqueeze(0) # (1, B, H)
        c0 = torch.zeros_like(h0)                          # (1, B, H)

        out, _ = self.lstm(dec_in, (h0, c0))  # out: (B, T, H)
        x_hat = self.hidden_to_out(out)        # (B, T, F)
        return x_hat
    
class VAE_LSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int=128, z_dim: int=32, num_layers: int=1, dropout: float=0.1):
        super().__init__()
        self.encoder = EncoderLSTM(in_dim, hidden_size, num_layers, z_dim, dropout)
        self.decoder = DecoderLSTM(in_dim, hidden_size, num_layers, z_dim, dropout)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        # x: (B, T, in_dim), mask: (B, T)
        mu, logvar = self.encoder(x, mask)          # (B, z_dim), (B, z_dim)
        std = torch.exp(0.5 * logvar)              # (B, z_dim)
        eps = torch.randn_like(std)                 # (B, z_dim)
        z = mu + eps * std                           # reparameterization trick
        x_hat = self.decoder(x, z)                  # (B, T, in_dim)
        return x_hat, mu, logvar
    
