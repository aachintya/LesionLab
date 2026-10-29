"""Atomic artifacts, immutable references and experiment identity."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import zipfile
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_identity():
    src = Path(__file__).resolve().parents[1]
    files = [*Path(__file__).parent.glob('*.py'), src/'scripts/binary_study.py',
             src/'components/preprocessing.py', src/'components/image_cache.py',
             src/'models/efficientnet_v2_image.py']
    return digest({str(p.relative_to(src)): sha(p) for p in sorted(files)})


def verify_reference(reference):
    root = Path(reference).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('format') != 'binary-reference-v1':
        raise ValueError('Not a binary study reference bundle')
    for name, expected in manifest['sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Immutable reference changed or is missing: {name}')
    return manifest


def runtime_versions():
    from importlib.metadata import version
    import platform
    return {'python':platform.python_version(),**{name:version(name) for name in
            ('torch','torchvision','numpy','pandas','scikit-learn','scipy','Pillow','joblib')}}


def initialize(output, reference):
    root, ref = Path(output).resolve(), Path(reference).resolve()
    if (root == Path.cwd().resolve() or 'baselines' in root.parts or
        root.is_relative_to(ref) or ref.is_relative_to(root) or
        ('src' in root.parts and 'checkpoints' in root.parts)):
        raise ValueError('Outputs must be isolated from the reference, baseline and app checkpoints')
    identity = {'format': 'binary-study-v1', 'reference_sha256': sha(ref/'manifest.json'),
                'source_sha256': source_identity(), 'environment': runtime_versions()}
    marker = root/'identity.json'
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('Run source/reference identity changed; use a new experiment directory')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('Refusing to adopt a nonempty unrelated output directory')
        root.mkdir(parents=True, exist_ok=True); write_json(marker, identity)
    verify_reference(ref)
    return root


def verify_lock(root):
    root = Path(root)
    lock = json.loads((root/'lock.json').read_text())
    if lock['source_sha256'] != source_identity():
        raise ValueError('Source changed after calibration lock')
    for name, expected in lock['sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Locked artifact changed: {name}')
    return lock


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def choose_device(name='auto'):
    import torch
    if name == 'auto': return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Enable a Kaggle GPU before the long run.')
    return device


def package(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.is_relative_to(root): raise ValueError('Artifact ZIP must be outside the run')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name+'.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(root.rglob('*')):
            rel = p.relative_to(root)
            if p.is_file() and not p.is_symlink() and 'cache' not in rel.parts and p.suffix != '.tmp':
                z.write(p, rel)
    os.replace(temp, destination)
    print(f'Saved resumable artifacts: {destination} ({destination.stat().st_size/1e6:.1f} MB)', flush=True)
    return destination
