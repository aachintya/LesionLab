"""Train images concurrently on two CUDA GPUs, then metadata on CPU."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def run_group(commands, log_dir):
    processes, handles = [], []
    try:
        for name, command in commands:
            log_path = log_dir / f"{name}.log"
            handle = log_path.open("a", buffering=1)
            handles.append(handle)
            print(f"Starting {name}; log: {log_path}", flush=True)
            processes.append((name, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                                     env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})))
        last_update = 0.0
        while any(p.poll() is None for _, p in processes):
            for name, process in processes:
                if process.poll() not in (None, 0):
                    tail = (log_dir / f"{name}.log").read_text()[-6000:]
                    raise RuntimeError(f"{name} failed (exit {process.returncode}):\n{tail}")
            if time.monotonic() - last_update > 30:
                for name, _ in processes:
                    lines = (log_dir / f"{name}.log").read_text().splitlines()
                    print(f"{name}: " + (lines[-1] if lines else "initializing"), flush=True)
                last_update = time.monotonic()
            time.sleep(1)
        for name, process in processes:
            if process.returncode:
                raise RuntimeError((log_dir / f"{name}.log").read_text()[-6000:])
            print(f"Finished {name}", flush=True)
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2, help="Workers per image process; use 1 on CPU-limited hosts")
    parser.add_argument("--clinical-epochs", type=int, default=30)
    parser.add_argument("--dermoscopic-epochs", type=int, default=20)
    parser.add_argument("--metadata-epochs", type=int, default=20)
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parents[2] / "logs")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. In Kaggle Settings enable the GPU accelerator before running training.")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("train_model.py")
    jobs = []
    parallel = args.parallel and torch.cuda.device_count() >= 2
    for index, modality in enumerate(("clinical", "dermoscopic")):
        command = [sys.executable, "-u", str(script), "--csv", str(args.csv), "--modality", modality,
                   "--epochs", str(getattr(args, f"{modality}_epochs")), "--batch-size", str(args.batch_size),
                   "--num-workers", str(args.num_workers), "--cache-dir", str(args.cache_dir),
                   "--device", f"cuda:{index if parallel else 0}", "--amp", "--channels-last"]
        if args.resume:
            command.append("--resume")
        jobs.append((modality, command))
    if parallel:
        run_group(jobs, args.log_dir)
    else:
        for job in jobs:
            run_group([job], args.log_dir)
    metadata = [sys.executable, "-u", str(script), "--csv", str(args.csv), "--modality", "metadata",
                "--epochs", str(args.metadata_epochs), "--device", "cpu"]
    if args.resume:
        metadata.append("--resume")
    run_group([("metadata", metadata)], args.log_dir)


if __name__ == "__main__":
    main()
