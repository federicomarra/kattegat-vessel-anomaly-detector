import torch
import torch.nn.functional as F

def sequence_loss_fn(
    recon_batch: torch.Tensor,  # (B, T, F)
    x_batch: torch.Tensor,      # (B, T, F)
) -> torch.Tensor:
    """
    Returns loss per sequence: shape (B,).
    Here you can easily change the definition (MSE, MAE, weights, etc.).
    """
    mse = F.mse_loss(recon_batch, x_batch, reduction="none")  # (B, T, F)
    # mean on time and feature, keeping batch separate
    seq_error = mse.mean(dim=(1, 2))  # (B,)
    return seq_error


def sequence_loss_fn2(recon_batch, x_batch, lambda_vel: float = 0.5):
    # base reconstruction
    mse = F.mse_loss(recon_batch, x_batch, reduction="none")  # (B, T, F)
    rec_err = mse.mean(dim=(1, 2))  # (B,)

    # temporal differences (approx velocity) on first 2 features (lon, lat)
    vel_true = x_batch[:, 1:, :2] - x_batch[:, :-1, :2]      # (B, T-1, 2)
    vel_recon = recon_batch[:, 1:, :2] - recon_batch[:, :-1, :2]

    vel_mse = (vel_true - vel_recon).pow(2).mean(dim=(1, 2))  # (B,)

    seq_error = rec_err + lambda_vel * vel_mse
    return seq_error

def sequence_loss_fn3(recon_batch, x_batch,
                     w_pos=4.0, w_sog=1.0, w_cog=1.0,
                     lambda_vel=0.5):
    """
    x / recon: (B, T, 4) = [lon, lat, SOG, COG]
    """
    # ---- errore per-feature ----
    pos_err = (recon_batch[..., 0:2] - x_batch[..., 0:2]).pow(2)      # (B,T,2)
    sog_err = (recon_batch[..., 2]   - x_batch[..., 2]).pow(2)        # (B,T)
    cog_err = (recon_batch[..., 3]   - x_batch[..., 3]).pow(2)        # (B,T)

    pos_err = pos_err.mean(dim=2)   # (B,T)
    sog_err = sog_err
    cog_err = cog_err

    rec_err = (w_pos * pos_err + w_sog * sog_err + w_cog * cog_err).mean(dim=1)  # (B,)

    # ---- termine sulle velocità (dal lon/lat assoluti) ----
    vel_true  = x_batch[:, 1:, 0:2] - x_batch[:, :-1, 0:2]           # (B,T-1,2)
    vel_recon = recon_batch[:, 1:, 0:2] - recon_batch[:, :-1, 0:2]
    vel_mse   = (vel_true - vel_recon).pow(2).mean(dim=(1, 2))       # (B,)

    seq_error = rec_err + lambda_vel * vel_mse
    return seq_error
