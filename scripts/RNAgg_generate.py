import sys
import os
import argparse
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



def generate_from_decoder(model, z, device):
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
    # parser.add_argument('--out_dir', default=None, help='output directory (defaults to results/ under project root)')
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
        # determine emb_dir: explicit or inferred from model parent/latents
        if args.emb_file:
            emb_dir = args.emb_file
        else:
            model_parent = os.path.dirname(os.path.abspath(args.model))
            emb_dir = os.path.join(model_parent, 'latents')

        mean_path = os.path.join(emb_dir, 'latents_mean.npy')
        var_path = os.path.join(emb_dir, 'latents_var.npy')

        means = np.load(mean_path)
        vars_ = np.load(var_path)
        means_t = torch.from_numpy(means).float().to(device)
        vars_t = torch.from_numpy(vars_).float().to(device)

        m = means_t.size(0)
        n = int(args.n)
        if m >= n:
            idx_np = np.random.choice(m, n, replace=False)
        else:
            idx_np = np.random.choice(m, n, replace=True)

        # safe indexing using torch LongTensor on device
        idx_t = torch.from_numpy(idx_np.astype(np.int64)).long().to(device)
        sel_means = means_t.index_select(0, idx_t)
        sel_vars = vars_t.index_select(0, idx_t)

        # detect whether sel_vars is logvar (median<0 heuristic) or variance
        if float(sel_vars.median()) < 0:
            sel_std = torch.exp(0.5 * sel_vars)
        else:
            sel_std = torch.sqrt(torch.clamp(sel_vars, min=1e-10))

        eps = torch.randn_like(sel_means)
        z = sel_means + eps * sel_std
    else:
        n = int(args.n)
        z = torch.normal(mean=0.0, std=1.0, size=(n, d_rep), device=device).float()

    # decode
    seqs = generate_from_decoder(model, z, device)

    # save sequences
    family = args.family if args.family else os.path.splitext(os.path.basename(args.model))[0]
    if args.from_emb:
        # save into the parent directory of the latents folder (e.g., data/RF00005)
        save_dir = os.path.abspath(os.path.dirname(emb_dir))
    else:
        # default results directory under project root
        save_dir = os.path.join(_project_root, f'data/{family}')
        save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    out_fname = os.path.join(save_dir, f"{family}.txt")

    with open(out_fname, 'w') as f:
        for s in seqs:
            f.write(s + "\n")

    print(f"Saved {len(seqs)} sequences to {out_fname}")
