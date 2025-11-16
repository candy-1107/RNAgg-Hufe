import sys
import os
import argparse
import pickle
import numpy as np

# ensure repo scripts on path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn.functional as F
import RNAgg_VAE

NUC_LETTERS = list('ACGU-x')
SINGLE_LABELS = ["A", "U", "G", "C", "gap"]
PAIR_LABELS = ["A-U", "U-A", "G-C", "C-G", "G-U", "U-G"]
ALL_LABELS = SINGLE_LABELS + PAIR_LABELS


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def matrices_to_seq_struct_from_probs(mat_probs, threshold=0.5):
    """
    mat_probs: numpy array shape (11, L, L) with values in [0,1]
    returns (seq_str, struct_str)
    """
    C, L1, L2 = mat_probs.shape
    assert C == len(ALL_LABELS) and L1 == L2
    L = L1
    seq = ['N'] * L
    struct = ['.'] * L
    # pair channels: indices 5..10 for i<j
    for i in range(L):
        for j in range(i+1, L):
            pair_vals = mat_probs[len(SINGLE_LABELS):, i, j]
            max_idx = int(np.argmax(pair_vals))
            max_val = float(pair_vals[max_idx])
            if max_val >= threshold:
                pair_label = PAIR_LABELS[max_idx]
                b1, b2 = pair_label.split('-')
                if seq[i] == 'N' or seq[i] == b1:
                    seq[i] = b1
                if seq[j] == 'N' or seq[j] == b2:
                    seq[j] = b2
                struct[i] = '('
                struct[j] = ')'
    # diagonal singles
    for i in range(L):
        if seq[i] != 'N':
            continue
        diag_vals = mat_probs[:len(SINGLE_LABELS), i, i]
        max_idx = int(np.argmax(diag_vals))
        max_val = float(diag_vals[max_idx])
        if max_val >= threshold:
            label = SINGLE_LABELS[max_idx]
            seq[i] = '-' if label == 'gap' else label
        else:
            seq[i] = 'N'
    return ''.join(seq), ''.join(struct)


def generate_from_conv_checkpoint(model, z, device, threshold=0.5):
    model.to(device)
    model.eval()
    with torch.no_grad():
        conv_shape = getattr(model, 'enc_shape', None)
        if conv_shape is None:
            raise RuntimeError('Conv_VAE missing enc_shape; ensure model initialized with correct input_shape')
        # decode: model.decode expects conv_shape
        y = model.decode(z.to(device), conv_shape)
        y_cpu = y.detach().cpu().numpy()
    results = []
    B, C, H, W = y_cpu.shape
    # For each sample, build probability matrices:
    for b in range(B):
        mat = y_cpu[b]  # shape (11, H, W)
        # We'll build mat_probs:
        mat_probs = np.zeros_like(mat)
        L = mat.shape[1]
        # diagonal channels (first 5): apply softmax across channels at each diagonal position
        for i in range(L):
            diag_raw = mat[:5, i, i]
            diag_prob = softmax(diag_raw)
            mat_probs[:5, i, i] = diag_prob
        # pair channels (6 channels): apply sigmoid to raw values for each (i,j)
        pair_raw = mat[5:, :, :]
        pair_sig = sigmoid(pair_raw)
        # copy only upper triangle i<j
        for i in range(L):
            for j in range(i+1, L):
                mat_probs[5:, i, j] = pair_sig[:, i, j]
        # note: lower triangle ignored
        seq, ss = matrices_to_seq_struct_from_probs(mat_probs, threshold=threshold)
        results.append((seq, ss))
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate sequences from a Conv_VAE checkpoint and save to results/<family>.txt')
    parser.add_argument('--model', required=True, help='path to saved Conv_VAE model .pth (must contain input_shape and d_rep)')
    parser.add_argument('-n', type=int, default=10, help='number of samples to generate (ignored if --from-emb)')
    parser.add_argument('--from-emb', action='store_true', help='treat -n as path to pickled embeddings to decode')
    parser.add_argument('--emb-file', help='(optional) alternative explicit path to embedding pickle file when using --from-emb')
    parser.add_argument('--family', help='family name to use as output filename (defaults to model basename)')
    parser.add_argument('--device', default=None, help='device (cpu or cuda); if omitted auto-detect')
    parser.add_argument('--threshold', type=float, default=0.5, help='probability threshold for deciding bases/pairs')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}', file=sys.stderr)

    ckpt = torch.load(args.model, map_location='cpu')
    if 'input_shape' not in ckpt or 'model_state_dict' not in ckpt or 'd_rep' not in ckpt:
        raise RuntimeError('Checkpoint does not look like a Conv_VAE checkpoint (missing input_shape/d_rep/model_state_dict)')

    input_shape = tuple(ckpt['input_shape'])
    d_rep = int(ckpt['d_rep'])

    model = RNAgg_VAE.Conv_VAE(input_shape, latent_dim=d_rep, device=device).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # prepare z
    if args.from_emb:
        emb_path = args.emb_file if args.emb_file else str(args.n)
        if not os.path.exists(emb_path):
            raise RuntimeError(f'Embedding file not found: {emb_path}')
        with open(emb_path, 'rb') as f:
            z = pickle.load(f)
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32)
        if isinstance(z, list):
            z = torch.tensor(np.asarray(z), dtype=torch.float32)
        if not isinstance(z, torch.Tensor):
            raise RuntimeError('Loaded embedding is not a tensor/ndarray')
        # ensure device
        z = z.to(device)
        n_samples = z.size(0)
    else:
        n_samples = int(args.n)
        z = torch.randn(n_samples, d_rep, device=device)

    results = generate_from_conv_checkpoint(model, z, device, threshold=args.threshold)

    # decide family name
    family = args.family if args.family else os.path.splitext(os.path.basename(args.model))[0]
    out_dir = os.path.join(_project_root, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{family}.txt")

    with open(out_path, 'w', encoding='utf-8') as fout:
        for i, (seq, ss) in enumerate(results):
            fout.write(f'gen{i}\t{seq}\t{ss}\n')

    print(f'Wrote {len(results)} sequences to {out_path}', file=sys.stderr)

if __name__ == '__main__':
    main()
