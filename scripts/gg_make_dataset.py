import os
import argparse
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class MatrixDataset(Dataset):
    """Simple Dataset wrapping a list/array of matrices.
    Each item returned is a torch.FloatTensor of shape (C, H, W).
    """
    def __init__(self, arrays):
        # arrays: numpy array (N, C, H, W) or list of arrays
        if isinstance(arrays, np.ndarray):
            self.data = arrays
        else:
            self.data = np.stack(arrays, axis=0) if len(arrays) > 0 else np.empty((0,))

    def __len__(self):
        return 0 if self.data.size == 0 else self.data.shape[0]

    def __getitem__(self, idx):
        if self.data.size == 0:
            raise IndexError("Dataset is empty")
        x = self.data[idx]
        # ensure float32
        return torch.from_numpy(x.astype(np.float32))


def load_aggregated_npz(path):
    data = np.load(path, allow_pickle=True)
    # pick first array-like object found
    if hasattr(data, 'files') and len(data.files) > 0:
        key = data.files[0]
        arr = data[key]
    else:
        # fallback
        arr = data
    # ensure shape (N, C, H, W)
    arr = np.asarray(arr)
    # if shape like (N, H, W) add channel dim
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    return arr


def collect_families(base_dir):
    # find subdirs that contain aggregated_matrices.npz
    families = []
    for d in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, d)
        if not os.path.isdir(path):
            continue
        agg = os.path.join(path, 'aggregated_matrices.npz')
        if os.path.exists(agg):
            families.append((d, agg))
    return families


def save_torch(path, dataset_array):
    # dataset_array: numpy array (N, C, H, W)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(torch.from_numpy(dataset_array.astype(np.float32)), path)


def pad_array_to(arr, target_shape):
    # arr: (N, C, H, W)
    N, C, H, W = arr.shape
    tC, tH, tW = target_shape
    if (C, H, W) == (tC, tH, tW):
        return arr
    pad_c = tC - C
    pad_h = tH - H
    pad_w = tW - W
    pad_width = ((0, 0), (0, pad_c), (0, pad_h), (0, pad_w))
    padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
    return padded


def load_sequence_channels(seq_dir, require_square=True):
    files = [f for f in sorted(os.listdir(seq_dir)) if f.endswith('.npy')]
    if not files:
        return None, 'no_channel_files'
    mats = []
    sizes = []
    for fn in files:
        path = os.path.join(seq_dir, fn)
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as e:
            return None, f'load_error:{fn}:{e}'
        if arr.ndim != 2:
            return None, f'not_2d:{fn}'
        if require_square and arr.shape[0] != arr.shape[1]:
            return None, f'not_square:{fn}:{arr.shape}'
        mats.append(arr)
        sizes.append(arr.shape[0])
    if len(set(sizes)) != 1:
        return None, f'size_mismatch:{set(sizes)}'
    stacked = np.stack(mats, axis=0)  # (C, L, L)
    return stacked, None


def build_family_from_subdirs(family_dir, skip_on_error=True):
    # family_dir contains subdirs for each sequence (e.g., 0,1,2...)
    seq_dirs = [d for d in sorted(os.listdir(family_dir)) if os.path.isdir(os.path.join(family_dir, d))]
    seq_arrays = []
    errors = {}
    max_L = 0
    for sd in seq_dirs:
        sd_path = os.path.join(family_dir, sd)
        # skip the aggregated file if present as a file
        if sd == 'aggregated_matrices.npz':
            continue
        arr, err = load_sequence_channels(sd_path)
        if err:
            errors[sd] = err
            if skip_on_error:
                continue
            else:
                raise RuntimeError(f'Error loading {sd_path}: {err}')
        seq_arrays.append(arr)
        if arr.shape[1] > max_L:
            max_L = arr.shape[1]
    if not seq_arrays:
        return None, errors
    # pad each (C,L,L) to (C, max_L, max_L)
    aligned = []
    for arr in seq_arrays:
        C, L, _ = arr.shape
        if L == max_L:
            aligned.append(arr)
        elif L < max_L:
            pad = ((0,0),(0, max_L-L),(0, max_L-L))
            padded = np.pad(arr, pad_width=pad, mode='constant', constant_values=0)
            aligned.append(padded)
        else:
            trimmed = arr[:, :max_L, :max_L]
            aligned.append(trimmed)
    dataset = np.stack(aligned, axis=0)  # (N, C, L, L)
    return dataset, errors


def main(args):
    out_data_dir = args.out_data_dir
    os.makedirs(out_data_dir, exist_ok=True)

    # process only aligned matrices
    for typ in ('matrices_aligned',):
        base_dir = os.path.join(args.output_dir, typ)
        if not os.path.isdir(base_dir):
            print(f"Directory for {typ} not found: {base_dir}. Skipping.")
            continue

        families = collect_families(base_dir)
        if args.families:
            families = [f for f in families if f[0] in args.families]

        combined_list = []
        print(f"Found {len(families)} families in {base_dir}")
        for fam_name, agg_path in families:
            family_dir = os.path.dirname(agg_path)
            # try building from per-sequence subdirs first
            dataset, errors = build_family_from_subdirs(family_dir, skip_on_error=True)
            if dataset is None:
                # fallback to aggregated file
                try:
                    arr = load_aggregated_npz(agg_path)
                except Exception as e:
                    print(f"Failed to load {agg_path}: {e}")
                    continue
                if arr.size == 0:
                    print(f"No data in {agg_path}")
                    continue
                # ensure arr shaped (N, C, H, W); if 2D treat as single-channel single-sample
                if arr.ndim == 2:
                    arr = arr[None, None, :, :]
                elif arr.ndim == 3:
                    arr = arr[:, None, :, :]
                dataset = arr
            # save per-family
            fam_out = os.path.join(out_data_dir, f"{fam_name}.{typ}.pt")
            save_torch(fam_out, dataset)
            print(f"Saved family dataset: {fam_out} (n={dataset.shape[0]}, shape={dataset.shape[1:]})")
            combined_list.append(dataset)

        # save combined (pad families to common C,H,W before concat)
        if combined_list:
            Cs = [a.shape[1] for a in combined_list]
            Hs = [a.shape[2] for a in combined_list]
            Ws = [a.shape[3] for a in combined_list]
            target = (max(Cs), max(Hs), max(Ws))
            padded_list = [pad_array_to(a, target) for a in combined_list]
            combined = np.concatenate(padded_list, axis=0)
            combined_out = os.path.join(out_data_dir, f"all.{typ}.pt")
            save_torch(combined_out, combined)
            print(f"Saved combined dataset: {combined_out} (n={combined.shape[0]}, shape={combined.shape[1:]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build datasets from aggregated matrix files')
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), '..', 'output'),
                        help='path to output (project) folder that contains matrices_aligned/matrices_unaligned')
    parser.add_argument('--out-data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                        help='where to save .pt dataset files')
    parser.add_argument('--families', nargs='*', help='optional family names to include (e.g. RF00001)')
    args = parser.parse_args()
    main(args)
