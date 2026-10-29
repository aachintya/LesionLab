"""Atomic, portable resume files for this project's own trusted training outputs."""
import hashlib
import json
from pathlib import Path
import torch

from utils.training import atomic_torch_save, rng_state, restore_rng


def dataset_hash(csv_path):
    return hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()


def restore_training(path, model, optimizer, scheduler, scaler, config, resume):
    state = {"epoch": -1, "best_score": -1.0, "epochs_no_improve": 0,
             "history": [], "complete": False}
    if not resume or not path.exists():
        return state
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("config") != config:
        raise ValueError(f"Resume configuration/data differ from {path}. Restore the original settings or start a fresh run without --resume.")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    if scaler is not None and saved.get("scaler"):
        scaler.load_state_dict(saved["scaler"])
    restore_rng(saved.get("rng"))
    print(f"Restored {path.name}: epoch {saved['epoch'] + 1}, complete={saved['complete']}", flush=True)
    return saved


def save_training(path, state, model, optimizer, scheduler, scaler, config):
    state.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), scaler=scaler.state_dict() if scaler is not None else {},
                 config=config, rng=rng_state())
    atomic_torch_save(state, path)
    summary_path = path.with_suffix(".json")
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({k: state[k] for k in
                                     ("epoch", "best_score", "epochs_no_improve", "history", "complete", "config")}, indent=2))
    temporary.replace(summary_path)
