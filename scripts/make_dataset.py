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
    """Load aggregated_matrices.npz and return array shaped (N, C, H, W).

    The aggregated npz created by preprocessing stores 11 separate keys (labels),
    each an (H, W) matrix. We stack them along channel dimension C, and add a
    batch dimension N=1, yielding (1, 11, H, W).

    If the file format differs (already stacked), we try to coerce into (N,C,H,W).
    """
    data = np.load(path, allow_pickle=True)

    # Typical case: multiple named arrays per label
    if hasattr(data, 'files') and len(data.files) > 0:
        files = list(data.files)
        # Prefer canonical label order if present
        canonical = ["A", "U", "G", "C", "gap",
                     "A-U", "U-A", "G-C", "C-G", "G-U", "U-G"]

        if all(lbl in files for lbl in canonical):
            keys = canonical
        else:
            # Fallback: keep the original order in the npz
            keys = files

        mats = []
        for k in keys:
            mat = np.asarray(data[k])
            if mat.ndim != 2:
                # If someone saved differently, try to coerce later
                mats = None
                break
            mats.append(mat)

        if mats is not None and len(mats) > 0:
            # (C, H, W)
            arr_c_hw = np.stack(mats, axis=0)
            # (N=1, C, H, W)
            return arr_c_hw[None, ...]

        # Fallback path when arrays are already stacked differently
        # Try common shapes and coerce to (N,C,H,W)
        # e.g., one key already holds (C,H,W) or (N,C,H,W)
        first = np.asarray(data[files[0]])
        if first.ndim == 3:
            # assume (C,H,W)
            return first[None, ...]
        if first.ndim == 4:
            # already (N,C,H,W)
            return first

    # Final fallback: treat the whole np.load result as array
    arr = np.asarray(data)
    if arr.ndim == 3:  # (C,H,W) -> add batch
        return arr[None, ...]
    if arr.ndim == 2:  # (H,W) -> assume single-channel, add C and N
        return arr[None, None, ...]
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


def main(args):
    out_data_dir = args.out_data_dir
    os.makedirs(out_data_dir, exist_ok=True)

    # process aligned and unaligned separately
    for typ in ('matrices_aligned', 'matrices_unaligned'):
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
            try:
                arr = load_aggregated_npz(agg_path)
            except Exception as e:
                print(f"Failed to load {agg_path}: {e}")
                continue
            if arr.size == 0:
                print(f"No data in {agg_path}")
                continue
            # save per-family
            fam_out = os.path.join(out_data_dir, f"{fam_name}.{typ}.pt")
            save_torch(fam_out, arr)
            print(f"Saved family dataset: {fam_out} (n={arr.shape[0]})")
            combined_list.append(arr)

        # save combined
        if combined_list:
            combined = np.concatenate(combined_list, axis=0)
            combined_out = os.path.join(out_data_dir, f"all.{typ}.pt")
            save_torch(combined_out, combined)
            print(f"Saved combined dataset: {combined_out} (n={combined.shape[0]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build datasets from aggregated matrix files')
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), '..', 'output'),
                        help='path to output (project) folder that contains matrices_aligned/matrices_unaligned')
    parser.add_argument('--out-data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                        help='where to save .pt dataset files')
    parser.add_argument('--families', nargs='*', help='optional family names to include (e.g. RF00001)')
    args = parser.parse_args()
    main(args)

