import sys
import os
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import copy
import importlib.util
import numpy as np
import matplotlib
# use non-interactive backend so saving PNGs works in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# helper to import local modules by filename if normal import fails
def import_local(module_name, filename):
    try:
        return __import__(module_name)
    except Exception:
        path = os.path.join(_this_dir, filename)
        if not os.path.exists(path):
            raise
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

# import local modules
RNAgg_VAE = import_local('RNAgg_VAE', 'RNAgg_VAE.py')
utils = import_local('utils_gg', 'utils_gg.py')

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

NUC_LETTERS = list('ACGU-x')
G_DIM = 11

# Labels consistent with preprocessing/_make_input_matrix.py
SINGLE_LABELS = ["A", "U", "G", "C", "gap"]
PAIR_LABELS = ["A-U", "U-A", "G-C", "C-G", "G-U", "U-G"]
ALL_LABELS = SINGLE_LABELS + PAIR_LABELS


def matrices_to_seq_struct(mat, threshold=0.5):
    """
    Convert a single sample's 11xLxL matrix (numpy or torch tensor) to sequence and dot-bracket structure.
    mat: np.ndarray or torch.Tensor with shape (11, L, L)
    Returns (seq_str, struct_str)
    """
    if hasattr(mat, 'cpu'):
        mat = mat.cpu().numpy()
    mat = np.asarray(mat)
    C, L1, L2 = mat.shape
    assert C == len(ALL_LABELS) and L1 == L2
    L = L1
    seq = ['N'] * L
    struct = ['.'] * L

    # Process pair channels: original building set pairs only at (i,j) with i<j
    # We'll look at upper triangle i<j and pick the argmax channel for each (i,j)
    for i in range(L):
        for j in range(i+1, L):
            # find best channel among pair channels for (i,j)
            pair_vals = mat[len(SINGLE_LABELS):, i, j]  # shape (6,)
            max_idx = np.argmax(pair_vals)
            max_val = float(pair_vals[max_idx])
            if max_val >= threshold:
                pair_label = PAIR_LABELS[max_idx]  # e.g. 'A-U'
                b1, b2 = pair_label.split('-')
                # assign if not already assigned by another pair
                if seq[i] == 'N' or seq[i] == b1:
                    seq[i] = b1
                else:
                    # conflict: keep existing assignment
                    pass
                if seq[j] == 'N' or seq[j] == b2:
                    seq[j] = b2
                else:
                    pass
                struct[i] = '('
                struct[j] = ')'

    # Process singles (diagonal)
    for i in range(L):
        if seq[i] != 'N':
            continue
        diag_vals = mat[:len(SINGLE_LABELS), i, i]
        max_idx = np.argmax(diag_vals)
        max_val = float(diag_vals[max_idx])
        if max_val >= threshold:
            label = SINGLE_LABELS[max_idx]
            if label == 'gap':
                seq[i] = '-'
            else:
                seq[i] = label
        else:
            seq[i] = 'N'

    seq_str = ''.join(seq)
    struct_str = ''.join(struct)
    return seq_str, struct_str


def build_dataloader_from_pt(pt_path, batch_size, n_layers=3):
    t = torch.load(pt_path, map_location='cpu')
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(np.asarray(t), dtype=torch.float32)
    # ensure shape N,C,H,W
    if t.ndim == 3:
        t = t.unsqueeze(1)
    N, C, H, W = t.shape
    factor = 2 ** n_layers
    target_H = ((H + factor - 1) // factor) * factor
    target_W = ((W + factor - 1) // factor) * factor
    pad_h = target_H - H
    pad_w = target_W - W
    if pad_h != 0 or pad_w != 0:
        # pad right and bottom: F.pad expects (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
        import torch.nn.functional as Fpad
        t = Fpad.pad(t, (0, pad_w, 0, pad_h))
        N, C, H, W = t.shape
    # If batch_size is None or <=0, use full dataset size (full-batch)
    if batch_size is None or (isinstance(batch_size, int) and batch_size <= 0):
        batch_size = t.size(0)
    dataset = TensorDataset(t, torch.full((t.size(0),), float('nan')))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, t.shape, t


def save_model(best_model, input_shape, best_epoch, model_path, args):
    if args.act_fname is None:
        model_type = 'conv_org'
    else:
        model_type = 'conv_act'
    nuc_yes_no = 'yes' if args.nuc_only else 'no'
    save_model_dict = {
        'model_state_dict': best_model.state_dict(),
        'd_rep': args.d_rep,
        'input_shape': input_shape,
        'best_epoch': best_epoch,
        'max_epoch': args.epoch,
        'lr': args.lr,
        'beta': args.beta,
        'type': model_type,
        'nuc_only': nuc_yes_no
    }
    # atomic save: write to a temp file in the same directory then atomically replace
    model_dir = os.path.dirname(os.path.abspath(model_path)) or '.'
    try:
        tmp_path = os.path.join(model_dir, os.path.basename(model_path) + '.tmp')
        torch.save(save_model_dict, tmp_path)
        # use os.replace for atomic rename on most platforms
        os.replace(tmp_path, model_path)
    except Exception:
        # fallback: try os.rename, then final fallback write directly
        try:
            os.rename(tmp_path, model_path)
        except Exception:
            torch.save(save_model_dict, model_path)
    print(f"Saved model to: {model_path}", file=sys.stderr)


def save_latents(model, tensor, save_dir, device, batch_size=100):
    """
    Save latent means and vars computed by the VAE for the provided tensor.
    Saves numpy files: latents_mean.npy and latents_var.npy in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()
    ds = TensorDataset(tensor, torch.full((tensor.size(0),), float('nan')))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    means_list = []
    vars_list = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, mean, var = model(x)
            means_list.append(mean.cpu().numpy())
            vars_list.append(var.cpu().numpy())
    means = np.concatenate(means_list, axis=0)
    vars_ = np.concatenate(vars_list, axis=0)
    mean_path = os.path.join(save_dir, 'latents_mean.npy')
    var_path = os.path.join(save_dir, 'latents_var.npy')
    np.save(mean_path, means)
    np.save(var_path, vars_)
    print(f"Saved latents (mean) to: {mean_path}", file=sys.stderr)
    print(f"Saved latents (var) to : {var_path}", file=sys.stderr)


def draw_loss(loss_fname_tuple: tuple, loss_list_tuple: tuple):
    for fname, loss_list in zip(loss_fname_tuple, loss_list_tuple):
        if not loss_list:
            continue
        plt.plot(loss_list)
        plt.savefig(fname)
        plt.close()
        print(f"Saved loss plot: {fname}", file=sys.stderr)


def checkArgs(args):
    if '/' in args.png_prefix:
        print(f"args.png_prefix({args.png_prefix}) should not contain '/'.", file=sys.stderr)
        exit(0)
    try:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=True)
            print(f"Created output directory: {args.out_dir}", file=sys.stderr)
        elif not os.path.isdir(args.out_dir):
            print(f"args.out_dir({args.out_dir}) exists but is not a directory.", file=sys.stderr)
            exit(1)
    except Exception as e:
        print(f"Cannot create or access args.out_dir({args.out_dir}): {e}", file=sys.stderr)
        exit(1)


def main(args: argparse.Namespace):
    checkArgs(args)
    model_path = os.path.join(args.out_dir, args.model_fname)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}", file=sys.stderr)

    # data-pt mode only
    if not args.data_pt:
        print("Error: --data-pt is required. Training supports only --data-pt mode.", file=sys.stderr)
        return
    if not os.path.exists(args.data_pt):
        print(f"Error: --data-pt file not found at {args.data_pt}", file=sys.stderr)
        return
    train_loader, shape, full_tensor = build_dataloader_from_pt(args.data_pt, batch_size=None, n_layers=args.n_layers)
    B, C, H, W = shape
    input_shape = (C, H, W)
    print(f"Loaded .pt dataset: {B} samples, shape (C,H,W)={input_shape}")

    model = RNAgg_VAE.Conv_VAE(input_shape, latent_dim=args.d_rep, device=device).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__}, Parameters: {params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_model = None
    best_loss = float('inf')
    best_epoch = 0
    Loss_list, L1_list, L2_list = [], [], []

    recon_loss_fn = nn.MSELoss(reduction='sum')

    for epoch in range(args.epoch):
        loss_sum = 0.0
        L1_sum = 0.0
        L2_sum = 0.0
        n_batches = 0
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader, start=1):
            x = x.to(device)
            optimizer.zero_grad()
            y, mean, var = model(x)
            L1 = recon_loss_fn(y, x) / x.size(0)
            L2 = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean ** 2 - var, dim=1))
            loss = L1 + args.beta * L2
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            L1_sum += L1.item()
            L2_sum += L2.item()
            n_batches += 1
            if args.verbose and batch_idx % 10 == 0:
                print(f'Epoch {epoch+1} Batch {batch_idx}: batch_loss={loss.item():.4f}, L1={L1.item():.4f}, L2={L2.item():.4f}', file=sys.stderr)
        if n_batches == 0:
            print('No batches found (empty loader).', file=sys.stderr)
            break
        loss_mean = loss_sum / n_batches
        L1_mean = L1_sum / n_batches
        L2_mean = L2_sum / n_batches
        Loss_list.append(loss_mean)
        L1_list.append(L1_mean)
        L2_list.append(L2_mean)
        print(f'Epoch: {epoch+1}, loss: {loss_mean:.3f}, L1_recon: {L1_mean:.3f}, L2_KL: {L2_mean:.3f}', file=sys.stderr)
        # Use strict improvement to save and replace atomically when a new best model is found
        if loss_mean < best_loss:
            best_loss = loss_mean
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
            save_model(best_model, input_shape, best_epoch, model_path, args)
            if args.save_latent_dir:
                latent_dir = args.save_latent_dir
                if not os.path.isabs(latent_dir):
                    latent_dir = os.path.join(args.out_dir, latent_dir)
                save_latents(best_model, full_tensor, latent_dir, device, batch_size=args.s_bat)
        if args.save_ongoing > 0 and (epoch + 1) % args.save_ongoing == 0:
            # periodic save of the current best (already replaced if improved above)
            save_model(best_model, input_shape, best_epoch, model_path, args)
            if args.save_latent_dir:
                latent_dir = args.save_latent_dir
                if not os.path.isabs(latent_dir):
                    latent_dir = os.path.join(args.out_dir, latent_dir)
                save_latents(best_model, full_tensor, latent_dir, device, batch_size=args.s_bat)

    if best_model is not None:
        save_model(best_model, input_shape, best_epoch, model_path, args)
        if args.save_latent_dir:
            latent_dir = args.save_latent_dir
            if not os.path.isabs(latent_dir):
                latent_dir = os.path.join(args.out_dir, latent_dir)
            save_latents(best_model, full_tensor, latent_dir, device, batch_size=args.s_bat)
        print(f"The best model was obtained at epoch={best_epoch}")
    else:
        print('No model saved.', file=sys.stderr)

    Loss_png_names = ("Loss.png", "L1.png", "L2.png")
    Loss_png_names = [args.png_prefix + x for x in Loss_png_names]
    Loss_png_names = [os.path.join(args.out_dir, x) for x in Loss_png_names]
    draw_loss(Loss_png_names, (Loss_list, L1_list, L2_list))


    if best_model is not None and args.save_recon_dir is not None:
        recon_dir = args.save_recon_dir
        if not os.path.isabs(recon_dir):
            recon_dir = os.path.join(args.out_dir, recon_dir)
        save_reconstructions(best_model, full_tensor, recon_dir, device, batch_size=args.s_bat, threshold=args.recon_threshold)
        print(f"Reconstructions saved to: {recon_dir}", file=sys.stderr)
    else:
        print("No reconstructions saved. Set --save-recon-dir to save reconstructions.", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data-pt mode only: require .pt dataset path
    parser.add_argument('--data-pt', required=True, default=None, help='input .pt dataset file (required)')
    parser.add_argument('--epoch', type=int, default=500, help='maximum epoch')
    parser.add_argument('--s_bat', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float,  default=0.001, help='learning rate')
    parser.add_argument('--beta', type=float,  default=0.001, help='hyper parameter beta')
    parser.add_argument('--d_rep', type=int,  default=8, help='dimension of latent vector')
    parser.add_argument('--out_dir', default='./', help='output directory')
    parser.add_argument('--model_fname', default='model_RNAgg.pth', help='model file name')
    parser.add_argument('--png_prefix', default='', help='prefix of png files')
    parser.add_argument('--save_ongoing', default=0, type=int, help='save model and latent spage during training')
    parser.add_argument('--nuc_only', action='store_true', help='nucleotide only model')
    parser.add_argument('--act_fname', help='activity file name (not used in --data-pt mode)')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers for downsampling/upsampling')
    parser.add_argument('--save-latent-dir', dest='save_latent_dir', default=None, help='directory to save latent vectors (relative to out_dir if not absolute)')
    parser.add_argument('--save-recon-dir', dest='save_recon_dir', default=None, help='directory to save reconstructions (relative to out_dir if not absolute)')
    parser.add_argument('--recon-threshold', type=float, default=0.5, help='threshold for reconstruction sequence/structure')
    parser.add_argument('--verbose', action='store_true', help='print per-batch training logs')
    args = parser.parse_args()
    main(args)
