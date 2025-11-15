import os
import sys
import torch
import numpy as np
import glob
from torch.utils.data import TensorDataset, DataLoader

def readInput(fname: str):
    d_seq = {}
    d_ss  = {}
    with open(fname) as f:
        for line in f:
            line = line.replace('\n','')
            items = line.split(' ')
            sid, seq, ss = items[0:3]
            d_seq[sid] = seq
            d_ss[sid]  = ss
    return d_seq, d_ss

def get_token2idx(nuc_letters): # Letterと行番号の辞書を作成
    d = {}
    for i,x in enumerate(nuc_letters):
        d[x] = i
    return d

def readAct(fname):
    sid2act = {}
    with open(fname) as f:
        for line in f:
            line = line.replace('\n','')
            items = line.split() # 可変にしておく
            sid, act = items[0], items[1]
            act = float(act)
            #print(type(sid))
            #exit(0)
            sid2act[sid] = act
    return sid2act


class Dataset:
    def __init__(self, input_mat, sid_list=None, act_list=None): # input_data is tensor or numpy array
        # convert input_mat to torch tensor if needed
        if isinstance(input_mat, np.ndarray):
            self.data = torch.from_numpy(input_mat).float()
        elif isinstance(input_mat, torch.Tensor):
            self.data = input_mat.float()
        else:
            # try to coerce a list
            try:
                arr = np.stack(input_mat, axis=0)
                self.data = torch.from_numpy(arr).float()
            except Exception:
                raise RuntimeError('Unsupported input_mat type for Dataset')

        self.sid_list = sid_list if sid_list is not None else [str(i) for i in range(len(self.data))]
        if act_list is None:
            self.act_list = torch.full((len(self.data),), float('nan'))
        else:
            self.act_list = torch.tensor(act_list, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index], self.sid_list[index], self.act_list[index]
    
    def __len__(self):
        return len(self.data)


# --- new helper functions for loading per-family .pt/.npy files and building a DataLoader ---

def _load_pt_or_npy(path):
    """Load a .pt (torch saved tensor or dict), .pt.npy (numpy fallback) or .npy file and return a numpy array shaped (N,C,H,W)."""
    if path.endswith('.pt.npy') or path.endswith('.npy') and not path.endswith('.pt.npy'):
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f'Failed to load numpy file {path}: {e}')
        # if 2D -> (1,1,H,W), 3D -> (N,C,H,W) or (C,H,W) -> treat as single sample
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            # ambiguous: could be (C,H,W) or (N,H,W); assume (C,H,W) -> single sample
            C, H, W = arr.shape
            arr = arr[None, :, :, :]
        elif arr.ndim == 4:
            pass
        else:
            raise RuntimeError(f'Unsupported numpy array ndim {arr.ndim} in {path}')
        return arr.astype(np.float32)
    else:
        # assume .pt saved by torch
        try:
            data = torch.load(path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f'Failed to load .pt file {path}: {e}')
        # if data is tensor
        if isinstance(data, torch.Tensor):
            arr = data.cpu().numpy()
        elif isinstance(data, dict):
            # try pick first tensor-like
            tensor = None
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    tensor = v; break
                if isinstance(v, np.ndarray):
                    tensor = torch.from_numpy(v); break
            if tensor is None and 'dataset' in data and isinstance(data['dataset'], torch.Tensor):
                tensor = data['dataset']
            if tensor is None:
                raise RuntimeError(f'No tensor-like entry in .pt dict: {path}')
            arr = tensor.cpu().numpy()
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            raise RuntimeError(f'Unsupported .pt content type: {type(data)} for {path}')
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[None, :, :, :]
        elif arr.ndim == 4:
            pass
        else:
            raise RuntimeError(f'Unsupported array ndim {arr.ndim} in {path}')
        return arr.astype(np.float32)


def list_family_files(data_dir, patterns=None):
    """Return list of candidate family files (.pt, .pt.npy, .npy) under data_dir."""
    patterns = patterns or ['*.matrices_aligned.pt', '*.matrices_aligned.pt.npy', '*.pt', '*.pt.npy', '*.npy']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(data_dir, p)))
    # also consider family directories containing .npy files (each seq as separate file)
    # If directory contains subdirs, each subdir is likely a family; handle only flat files here.
    files = sorted(set(files))
    return files


def build_dataloader_from_data_dir(data_dir, batch_size=100, shuffle=True, families=None):
    """Load all family files under data_dir, pad to common shape and return (DataLoader, shape, full_tensor).
    shape returned is (N,C,H,W) where N is total samples across families.
    """
    if not os.path.isdir(data_dir):
        raise RuntimeError(f'data_dir not a directory: {data_dir}')
    # find candidate family files
    files = list_family_files(data_dir)
    # also accept per-family files named like RF00001.matrices_aligned.pt in directory
    # Optionally filter by families
    if families:
        files = [f for f in files if os.path.basename(f).split('.')[0] in families]
    if not files:
        # maybe the directory contains family subdirectories (each with many .npy files)
        # try scan subdirs and load aggregated file inside them if present
        subdirs = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
        for sd in subdirs:
            # pick any .pt or .npy inside
            cand = list_family_files(sd)
            if cand:
                files.extend(cand)
    if not files:
        raise RuntimeError(f'No family .pt/.npy files found in {data_dir}')

    arrays = []
    shapes = []
    for f in files:
        try:
            arr = _load_pt_or_npy(f)
        except Exception as e:
            print(f'Warning: skipping {f} due to load error: {e}', file=sys.stderr)
            continue
        arrays.append(arr)
        shapes.append(arr.shape[1:])  # (C,H,W)
    if not arrays:
        raise RuntimeError('No valid arrays loaded from data_dir')
    # determine target shape
    max_C = max(s[0] for s in shapes)
    max_H = max(s[1] for s in shapes)
    max_W = max(s[2] for s in shapes)

    padded_list = []
    for arr in arrays:
        N, C, H, W = arr.shape
        if (C, H, W) == (max_C, max_H, max_W):
            padded_list.append(arr)
        else:
            pad_c = max_C - C
            pad_h = max_H - H
            pad_w = max_W - W
            pad_width = ((0,0), (0,pad_c), (0,pad_h), (0,pad_w))
            padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
            padded_list.append(padded)
    full = np.concatenate(padded_list, axis=0)
    # convert to torch tensor
    full_tensor = torch.from_numpy(full).float()
    N, C, H, W = full_tensor.shape
    ds = TensorDataset(full_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader, (N, C, H, W), full_tensor


def build_dataloader_from_pt(pt_path, batch_size=100, shuffle=True):
    """Load a single .pt/.npy dataset file and return (DataLoader, shape, full_tensor).
    Accepts torch-saved tensors, numpy arrays, dicts with tensor/array, or a Dataset-like object.
    Returns loader, (N,C,H,W), full_tensor(torch.Tensor).
    """
    if not os.path.exists(pt_path):
        raise RuntimeError(f".pt file not found: {pt_path}")
    # Use the internal loader to support multiple file types
    try:
        arr = _load_pt_or_npy(pt_path)
    except Exception:
        # fallback to torch.load path for .pt files that _load_pt_or_npy may not handle
        try:
            data = torch.load(pt_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f'Failed to load file {pt_path}: {e}')
        if hasattr(data, '__len__') and hasattr(data, '__getitem__') and not isinstance(data, (torch.Tensor, dict)):
            sample = data[0]
            if isinstance(sample, (list, tuple)):
                tensor = sample[0]
            else:
                tensor = sample
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            full_tensor = tensor.float()
        elif isinstance(data, dict):
            tensor = None
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    tensor = v; break
                if isinstance(v, np.ndarray):
                    tensor = torch.from_numpy(v); break
            if tensor is None and 'dataset' in data and isinstance(data['dataset'], torch.Tensor):
                tensor = data['dataset']
            if tensor is None:
                raise RuntimeError('No tensor-like entry found in .pt dict')
            full_tensor = tensor.float()
        elif isinstance(data, np.ndarray):
            full_tensor = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            full_tensor = data.float()
        else:
            raise RuntimeError(f'Unsupported .pt content type: {type(data)}')
    else:
        # arr is numpy array (N,C,H,W) or convertible
        full_tensor = torch.from_numpy(arr).float()

    # Ensure shape is (N, C, H, W)
    if full_tensor.ndim == 2:
        full_tensor = full_tensor.unsqueeze(0).unsqueeze(0)
    elif full_tensor.ndim == 3:
        full_tensor = full_tensor.unsqueeze(1)
    elif full_tensor.ndim == 4:
        pass
    else:
        raise RuntimeError(f'Unsupported tensor ndim: {full_tensor.ndim}')

    N, C, H, W = full_tensor.shape
    if full_tensor.dtype != torch.float32:
        full_tensor = full_tensor.float()
    ds = TensorDataset(full_tensor)
    drop_last = True if batch_size and batch_size > 1 else False
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader, (N, C, H, W), full_tensor

