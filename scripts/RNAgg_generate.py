import sys
import os
import argparse
import pickle
import numpy as np
import torch
import RNAgg_VAE

# ensure repo scripts on path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)



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


def generate_from_decoder(model, z, device):
    """
    Decode latent vectors using model.decoder(z) and convert to sequences.
    Steps (as requested):
      - call model.decoder(z) -> y with shape (n, 11, h, w)
      - permute to (n, h, 11, w)
      - sum over last dim -> (n, h, 11)
      - softmax over class dim (11) and argmax -> indices (n, h)
      - map indices to ALL_LABELS and then to single-letter bases where appropriate
    Returns list of sequence strings (one per sample).
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        # ensure z is on the right device
        z_t = z.to(device)
        # prefer decoder attribute, fall back to decode if necessary
        if hasattr(model, 'decoder'):
            y = model.decoder(z_t)
        elif hasattr(model, 'decode'):
            # try calling decode without conv shape
            try:
                y = model.decode(z_t)
            except Exception:
                raise RuntimeError('Model does not support decoder(z) and decode(z) failed; cannot decode with this model')
        else:
            raise RuntimeError('Model has no decoder/decode attribute')
        y_cpu = y.detach().cpu()
    # y_cpu expected shape (n, 11, h, w)
    if y_cpu.dim() != 4:
        raise RuntimeError(f'decoder output expected 4D tensor (n,11,h,w) but got shape {tuple(y_cpu.shape)}')
    n, c, h, w = y_cpu.shape
    if c != len(ALL_LABELS):
        # allow some flexibility but warn
        raise RuntimeError(f'decoder output channel dimension {c} != expected {len(ALL_LABELS)}')
    # permute to (n, h, 11, w)
    y_perm = y_cpu.permute(0, 2, 1, 3).contiguous()
    # sum over last dim -> (n, h, 11)
    y_sum = y_perm.sum(dim=3)
    # softmax over class dim (2)
    probs = torch.nn.functional.softmax(y_sum, dim=2)
    # argmax to get indices (n, h)
    idx = torch.argmax(probs, dim=2)
    idx_np = idx.cpu().numpy()
    results = []
    for i in range(n):
        labels = []
        for j in range(h):
            k = int(idx_np[i, j])
            label = ALL_LABELS[k]
            # convert to a single character for sequence: if single label, map 'gap'->'-'
            if label in SINGLE_LABELS:
                if label == 'gap':
                    labels.append('-')
                else:
                    labels.append(label)
            else:
                # pair label like 'A-U' -> take first base before '-' as the residue at this position
                first = label.split('-')[0]
                labels.append(first)
        seq_str = ''.join(labels)
        results.append(seq_str)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sequences from a Conv_VAE checkpoint and save to results/<family>.txt')
    parser.add_argument('--model', required=True, help='path to saved Conv_VAE model .pth (must contain input_shape and d_rep)')
    parser.add_argument('-n', type=int, default=10, help='number of samples to generate (ignored if --from-emb)')
    parser.add_argument('--from-emb', action='store_true', help='treat -n as path to pickled embeddings to decode')
    parser.add_argument('--emb-file', help='(optional) alternative explicit path to embedding pickle file when using --from-emb')
    parser.add_argument('--family', help='family name to use as output filename (defaults to model basename)')
    parser.add_argument('--device', default=None, help='device (cpu or cuda); if omitted auto-detect')
    parser.add_argument('--threshold', type=float, default=0.5, help='probability threshold for deciding bases/pairs')
    parser.add_argument('--out_dir', default=None, help='output directory (defaults to results/ under project root)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # load checkpoint onto chosen device (so tensors like input_shape, etc. are available)
    ckpt = torch.load(args.model, map_location=device)
    # support older checkpoints that only stored max_len
    if 'input_shape' not in ckpt:
        max_len = ckpt.get('max_len')
        if max_len is None:
            raise RuntimeError('Checkpoint missing input shape and max_len; cannot infer model input size')
        input_C = ckpt.get('input_C', 11)
        input_shape = (input_C, max_len, max_len)
    else:
        input_shape = tuple(ckpt['input_shape'])

    if 'd_rep' not in ckpt or 'model_state_dict' not in ckpt:
        raise RuntimeError('Checkpoint does not look like a Conv_VAE checkpoint (missing d_rep/model_state_dict)')

    d_rep = int(ckpt['d_rep'])

    # always use Conv_VAE
    model = RNAgg_VAE.Conv_VAE(input_shape, latent_dim=d_rep, device=str(device)).to(device)
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

    # If embeddings were provided, use the decoder-based generation path
    if args.from_emb:
        seqs = generate_from_decoder(model, z, device)
        # decide family name and output directory
        family = args.family if args.family else os.path.splitext(os.path.basename(args.model))[0]
        out_dir = args.out_dir if args.out_dir else os.path.join(_project_root, 'results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{family}.txt")
        with open(out_path, 'w', encoding='utf-8') as fout:
            for i, seq in enumerate(seqs):
                fout.write(f'gen{i}\t{seq}\n')
        print(f'Wrote {len(seqs)} sequences to {out_path}', file=sys.stderr)
    else:
        results = generate_from_conv_checkpoint(model, z, device, threshold=args.threshold)
        # decide family name and output directory
        family = args.family if args.family else os.path.splitext(os.path.basename(args.model))[0]
        out_dir = args.out_dir if args.out_dir else os.path.join(_project_root, 'results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{family}.txt")
        with open(out_path, 'w', encoding='utf-8') as fout:
            for i, (seq, ss) in enumerate(results):
                fout.write(f'gen{i}\t{seq}\t{ss}\n')
        print(f'Wrote {len(results)} sequences to {out_path}', file=sys.stderr)