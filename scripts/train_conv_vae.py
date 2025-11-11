import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from RNAgg_VAE import Conv_VAE

# Try import MatrixDataset from make_dataset (optional)
try:
    from make_dataset import MatrixDataset  # noqa: F401
except Exception:
    MatrixDataset = None


def kl_divergence(mu, var):
    # KL(q(z|x) || p(z)) for diagonal Gaussian; mean over batch
    return -0.5 * torch.mean(torch.sum(1 + torch.log(var + 1e-8) - mu**2 - var, dim=1))


def main():
    parser = argparse.ArgumentParser(description='Train Conv-VAE on aggregated matrix datasets (.pt)')
    parser.add_argument('--data', required=True, help='Path to a .pt file (e.g., ../data/all.matrices_aligned.pt)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3, help='Conv down/up layers')
    parser.add_argument('--beta', type=float, default=1e-3, help='KL weight')
    parser.add_argument('--norm', choices=['none', 'global', 'channel'], default='channel',
                        help='Normalize targets to [0,1] for BCE: none | global max | per-channel max')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save', default='../models/conv_vae.pth', help='Where to save model state_dict')
    parser.add_argument('--bn-eval-when-batch1', action='store_true',
                        help='Set all BatchNorm layers to eval when current batch size is 1 (avoids BN error)')
    parser.add_argument('--no-bn', action='store_true', help='Remove all BatchNorm layers from model for very small datasets')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load tensor (N, C, H, W)
    x = torch.load(args.data)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    # Coerce shapes to (N,C,H,W)
    if x.ndim == 2:           # (H,W)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:         # (N,H,W) -> add channel dim
        x = x.unsqueeze(1)
    elif x.ndim != 4:
        raise ValueError(f'Expect 4D tensor (N,C,H,W), got shape {tuple(x.shape)} from {args.data}')

    N, C, H, W = x.shape

    # Normalize to [0,1] if needed for BCE targets
    x = x.float()
    if args.norm == 'global':
        mx = torch.amax(x)
        if mx > 0:
            x = x / mx
    elif args.norm == 'channel':
        # (N,C,1,1)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        x = torch.where(mx > 0, x / mx, x)

    # Dataloader
    # Using simple TensorDataset for now; MatrixDataset isn't required at this stage
    ds = TensorDataset(x)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # Model
    model = Conv_VAE(input_shape=(C, H, W), latent_dim=args.latent_dim, n_layers=args.layers, device=device).to(device)

    if args.no_bn:
        # Replace all BatchNorm layers with Identity to avoid issues with tiny batch sizes
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                parent = model
                *path, last = name.split('.')
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, torch.nn.Identity())
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = total_rec = total_kl = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            optim.zero_grad()
            # Handle BN with batch size 1
            if args.bn_eval_when_batch1 and batch.size(0) == 1:
                for m in model.modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                        m.eval()
            x_rec, mu, var = model(batch)
            # Reconstruction loss: per-pixel BCE for 0/1 matrices; MSE also works. We'll use BCEWithLogitsLoss if last layer raw logits
            # Our Conv_VAE outputs raw features without sigmoid; use BCEWithLogits for binary targets.
            rec = F.binary_cross_entropy_with_logits(x_rec, batch, reduction='mean')
            kl = kl_divergence(mu, var)
            loss = rec + args.beta * kl
            loss.backward()
            optim.step()

            total_loss += loss.item() * batch.size(0)
            total_rec  += rec.item()  * batch.size(0)
            total_kl   += kl.item()   * batch.size(0)
        print(f'Epoch {ep:03d} | loss={total_loss/N:.4f} rec={total_rec/N:.4f} kl={total_kl/N:.4f}')

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'input_shape': (C, H, W),
        'latent_dim': args.latent_dim,
        'n_layers': args.layers,
        'beta': args.beta,
    }, args.save)
    print(f'Saved model to {args.save}')


if __name__ == '__main__':
    main()
