import sys
import os
import argparse
import copy
import gc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import RNAgg_VAE
import utils_gg as utils

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)



SINGLE_LABELS = ["A", "U", "G", "C", "gap"]
PAIR_LABELS = ["A-U", "U-A", "G-C", "C-G", "G-U", "U-G"]
ALL_LABELS = SINGLE_LABELS + PAIR_LABELS

# NUC_LETTERS = list('ACGU-x')
# G_DIM = 11

def main(args: dict):

    model_path = os.path.join(args['out_dir'], args['model_fname'])

    # token2idx = utils.get_token2idx(NUC_LETTERS)
    # idx2token = {v: k for k, v in token2idx.items()}
    # print(token2idx)
    # print(idx2token)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, shape, full_tensor = utils.build_dataloader_from_pt(args['data_pt'], batch_size=args.get('s_bat', 100), shuffle=True)
    B, C, H, W = shape
    input_shape = (C, H, W)

    model = RNAgg_VAE.Conv_VAE(input_shape, latent_dim=args['d_rep'], n_layers=args.get('n_layers', 3), device=str(device)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.get('lr', 0.001))

    Loss_list, L1_list, L2_list = [], [], []
    L1_nuc_list, L1_pair_list = [], []
    best_loss = float('inf')
    best_state = None
    best_epoch = 0
    eps = 1e-8
    SINGLE_N = len(SINGLE_LABELS)

    for epoch in range(1, args.get('epoch') + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_L1_sum = 0.0
        epoch_L2_sum = 0.0
        epoch_L1_nuc_sum = 0.0
        epoch_L1_pair_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch[0]
            x = x.to(device)
            N = x.size(0)

            optimizer.zero_grad()
            x_rec, mu, var = model(x)
            z = model.reparameterize(mu, var)

        #loss：sid
            pred_single = x_rec[:, :SINGLE_N].diagonal(offset=0, dim1=2, dim2=3).permute(0, 1, 2)       #(N, SINGLE_N, L_pred)
            target_single = x[:, :SINGLE_N].diagonal(offset=0, dim1=2, dim2=3).permute(0, 1, 2)        #(N, SINGLE_N, L_target)
            L_pred = pred_single.size(2)
            L_target = target_single.size(2)
            Lmin = min(L_pred, L_target)
            if Lmin <= 0:
                raise RuntimeError(f'Invalid diagonal length: pred={L_pred}, target={L_target}')
            if L_pred != Lmin:
                pred_single = pred_single[:, :, :Lmin]
            if L_target != Lmin:
                target_single = target_single[:, :, :Lmin]
            target_idx = target_single.argmax(dim=1)  # (N, Lmin)
            true_prob = pred_single.gather(1, target_idx.unsqueeze(1)).squeeze(1)  # (N, Lmin)
            per_pos_loss = -torch.log(true_prob + eps)
            L1_nuc = per_pos_loss.sum(dim=1).mean()

            pred_pairs = x_rec[:, SINGLE_N:]
            target_pairs = x[:, SINGLE_N:]
            ph, pw = pred_pairs.size(2), pred_pairs.size(3)
            th, tw = target_pairs.size(2), target_pairs.size(3)
            mh, mw = min(ph, th), min(pw, tw)
            if mh <= 0 or mw <= 0:
                raise RuntimeError(f'Invalid pair spatial dims: pred=({ph},{pw}), target=({th},{tw})')
            if ph != mh or pw != mw:
                pred_pairs = pred_pairs[:, :, :mh, :mw]
            if th != mh or tw != mw:
                target_pairs = target_pairs[:, :, :mh, :mw]
            L1_pair = F.binary_cross_entropy(pred_pairs, target_pairs, reduction='sum') / N

            L1 = L1_nuc + L1_pair

        # loss：mu，var
            L2 = -0.5 * torch.mean(torch.sum(1 + torch.log(var + eps) - mu ** 2 - var, dim=1))

        #total loss
            loss = L1 + args.get('beta', 0.0) * L2
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_L1_sum += L1.item()
            epoch_L2_sum += L2.item()
            epoch_L1_nuc_sum += L1_nuc.item()
            epoch_L1_pair_sum += L1_pair.item()
            n_batches += 1

        loss_mean = epoch_loss_sum / n_batches
        L1_mean = epoch_L1_sum / n_batches
        L2_mean = epoch_L2_sum / n_batches
        L1_nuc_mean = epoch_L1_nuc_sum / n_batches
        L1_pair_mean = epoch_L1_pair_sum / n_batches

        Loss_list.append(loss_mean)
        L1_list.append(L1_mean)
        L2_list.append(L2_mean)
        L1_nuc_list.append(L1_nuc_mean)
        L1_pair_list.append(L1_pair_mean)

        print(f'Epoch {epoch}: loss={loss_mean:.6f}, L1={L1_mean:.6f}, L1_nuc={L1_nuc_mean:.6f}, L1_pair={L1_pair_mean:.6f}, L2={L2_mean:.6f}', file=sys.stderr)

        # save best model (deep copy state dict)
        if loss_mean < best_loss:
            best_loss = loss_mean
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            # use centralized save_model to persist best state
            save_model(best_state, input_shape, best_epoch, model_path, args)

        if args.get('save_ongoing') and args.get('save_ongoing') > 0 and epoch % args.get('save_ongoing') == 0:
            # periodic ongoing save (save current state dict with epoch in filename)
            cur_state = copy.deepcopy(model.state_dict())
            save_model(cur_state, input_shape, epoch, model_path + f'.epoch{epoch}', args)

        # try to release GPU cached memory at end of epoch to reduce fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


    if best_state is not None:
        # save final/best-state using centralized save_model
        save_model(best_state, input_shape, best_epoch, model_path, args)
        print(f"Saved best model to {model_path} (epoch {best_epoch})", file=sys.stderr)
    else:
        print('No model was saved (no training performed).', file=sys.stderr)

    # plotting: build filenames from png_prefix and out_dir, then delegate to draw_loss
    Loss_png_names = ("Loss.png", "L1.png", "L2.png")
    # prefix each with png_prefix (may be empty)
    Loss_png_names = [args.get('png_prefix', '') + x for x in Loss_png_names]
    # make full paths under out_dir
    Loss_png_names = [os.path.join(args['out_dir'], x) for x in Loss_png_names]
    draw_loss(Loss_png_names, (Loss_list, L1_list, L2_list))
    latent_png_name = os.path.join(args['out_dir'], args.get('png_prefix', '') + 'latent.png')
    print(f"Saved loss plots to {args['out_dir']}", file=sys.stderr)


def save_model(best_model, input_shape, best_epoch, model_path, args):
    if isinstance(best_model, dict):
        model_state = copy.deepcopy(best_model)
    else:
        model_state = copy.deepcopy(best_model.state_dict())

    save_model_dict = {
        'model_state_dict': model_state,
        'd_rep': args['d_rep'],
        'input_shape': input_shape,
        'best_epoch': best_epoch,
        'max_epoch': args.get('epoch'),
        'lr': args.get('lr'),
        'beta': args.get('beta'),
        'type': 'conv_org'
    }
    tmp_path = model_path + '.tmp'
    torch.save(save_model_dict, tmp_path)
    os.replace(tmp_path, model_path)


def draw_loss(loss_fname_tuple: tuple, loss_list_tuple: tuple):
    for fname, loss_list in zip(loss_fname_tuple, loss_list_tuple):
        plt.plot(loss_list)
        plt.savefig(fname)
        plt.close()


def save_latents(model, tensor, save_dir, device, batch_size=100):

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--act_fname', help='activity file name (not used in --data-pt mode)')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers for downsampling/upsampling')
    parser.add_argument('--save-latent-dir', dest='save_latent_dir', default=None, help='directory to save latent vectors (relative to out_dir if not absolute)')
    parser.add_argument('--save-recon-dir', dest='save_recon_dir', default=None, help='directory to save reconstructions (relative to out_dir if not absolute)')
    parser.add_argument('--recon-threshold', type=float, default=0.5, help='threshold for reconstruction sequence/structure')
    parser.add_argument('--verbose', action='store_true', help='print per-batch training logs')
    args = vars(parser.parse_args())
    main(args)
