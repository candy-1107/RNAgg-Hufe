import sys
import os
import argparse
import pickle
import numpy as np
import RNAgg_VAE
import utils_gg as utils
from torch.utils.data import DataLoader
import torch

sys.path.append(os.environ.get('HOME', '') + "/pyscript")
NUC_LETTERS = list('ACGU-x')
G_DIM=11

# helper: parse dot-bracket
def _parse_dot_bracket(ss: str):
    stack = []
    pairs = {}
    for i, ch in enumerate(ss):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if not stack:
                continue
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j
    return pairs

# helper: build binary channel matrices (N, C, L, L)
def build_binary_matrices(sid2seq, sid2ss, sid_list, input_C=None, nuc_only_flag='no'):
    SINGLE = ["A","U","G","C","gap"]
    PAIRS = ["A-U","U-A","G-C","C-G","G-U","U-G"]
    default_C = len(SINGLE) + len(PAIRS)
    C = input_C if input_C is not None else (default_C if nuc_only_flag != 'yes' else len(SINGLE))
    mats = []
    for sid in sid_list:
        seq = sid2seq.get(sid, '')
        ss = sid2ss.get(sid, '')
        L = len(seq)
        mat = np.zeros((C, L, L), dtype=np.float32)
        pairs = _parse_dot_bracket(ss)
        # pairs channels
        for i in range(L):
            if i in pairs:
                j = pairs[i]
                if i < j:
                    b1 = seq[i] if i < len(seq) else 'N'
                    b2 = seq[j] if j < len(seq) else 'N'
                    if b1 not in ['A','U','G','C']:
                        b1 = 'gap'
                    if b2 not in ['A','U','G','C']:
                        b2 = 'gap'
                    pair_label = f"{b1}-{b2}"
                    if pair_label in PAIRS and C >= len(SINGLE) + len(PAIRS):
                        ch = len(SINGLE) + PAIRS.index(pair_label)
                        mat[ch, i, j] = 1.0
                        # reverse orientation
                        rev = f"{b2}-{b1}"
                        rev_idx = len(SINGLE) + PAIRS.index(rev)
                        mat[rev_idx, j, i] = 1.0
        # single/diagonal channels
        for i in range(L):
            b = seq[i] if i < len(seq) else 'N'
            if b not in ['A','U','G','C']:
                b = 'gap'
            if b in SINGLE:
                idx = SINGLE.index(b)
                if idx < C:
                    mat[idx, i, i] = 1.0
        mats.append(mat)
    return np.stack(mats, axis=0)


def main(args):
    token2idx = utils.get_token2idx(NUC_LETTERS)
    idx2token = dict([y,x] for x,y in token2idx.items())
    word_size = len(NUC_LETTERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, file=sys.stderr)

    # load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    d_rep = checkpoint.get('d_rep')
    input_shape = checkpoint.get('input_shape')
    if input_shape is None:
        max_len = checkpoint.get('max_len')
        if max_len is None:
            raise RuntimeError('Checkpoint missing input shape and max_len; cannot infer model input size')
        input_C = 11
        input_shape = (input_C, max_len, max_len)

    # always use Conv_VAE (remove MLP variants)
    model = RNAgg_VAE.Conv_VAE(input_shape, latent_dim=d_rep, device=device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    sid2seq, sid2ss = utils.readInput(args.input)
    sid_list = list(sid2seq.keys())
    sid2act = dict([(sid, np.nan) for sid in sid_list])
    act_list = [sid2act[sid] for sid in sid_list]

    B_mat = build_binary_matrices(sid2seq, sid2ss, sid_list, input_C=input_shape[0], nuc_only_flag=checkpoint.get('nuc_only','no'))

    # pad to target H,W
    N, C, H, W = B_mat.shape
    tC, tH, tW = input_shape
    if H != tH or W != tW or C != tC:
        padded = np.zeros((N, tC, tH, tW), dtype=np.float32)
        for i in range(N):
            ccopy = min(C, tC)
            hcopy = min(H, tH)
            wcopy = min(W, tW)
            padded[i, :ccopy, :hcopy, :wcopy] = B_mat[i, :ccopy, :hcopy, :wcopy]
        B_mat = padded

    B_t = torch.tensor(B_mat, dtype=torch.float32)
    d = utils.Dataset(B_t, sid_list, act_list)
    train_dataloader = DataLoader(d, batch_size=args.s_bat, shuffle=False)
    sid_list_exec = []
    for x, t, v in train_dataloader:
        x = x.to(device)
        # Conv_VAE.forward returns x_rec, mu, var
        _, mean, var = model(x)
        z_bat = mean
        if 'z' in locals():
            z = torch.cat((z, z_bat), dim=0)
        else:
            z = z_bat
        print(z.shape, file=sys.stderr)
        sid_list_exec += t

    z_np = z.to('cpu').detach().numpy().copy()
    emb_inf = {"sid_list": sid_list_exec, "emb": z_np}
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(emb_inf, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file name')
    parser.add_argument('model', help='trained VAE model')
    parser.add_argument('out_pkl', help='output pkl file')
    parser.add_argument('--s_bat', type=int, default=100, help='batch size')
    args = parser.parse_args()

    main(args)
