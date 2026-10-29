"""Concurrent controlled runs and real GPU preflight with clean cancellation."""
from __future__ import annotations

from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from improved.data import paired_frame
from components.metadata_encoder import MetadataEncoder
from ji_adf.model import JIADFModel, PRECISION_POLICY, ji_adf_loss
from ji_adf.runtime import capture_provenance, output_directory, verify_reference, write_json
from utils.training import _autocast, pick_device

TRAIN_KEYS = ("initialization", "epochs", "freeze_epochs", "patience", "size", "batch_size", "accumulate",
              "head_lr", "backbone_lr", "weight_decay", "workers", "cpu_threads")


def run(args):
    reference = Path(args.reference).resolve()
    verify_reference(reference)
    directory = output_directory(args.output, reference)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before running the JI-ADF training jobs")
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants)
            or "ji_adf" not in args.variants):
        raise ValueError("Use unique seeds/variants and include the predeclared ji_adf primary")
    plan = {"primary": "ji_adf", "variants": args.variants, "seeds": args.seeds,
            "precision": PRECISION_POLICY,
            **{key: getattr(args, key) for key in TRAIN_KEYS}}
    plan_path = directory / "run_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Run plan changed. Keep the original settings or use a new output folder")
    write_json(plan_path, plan)
    capture_provenance(directory, reference)
    pending = [(variant, seed) for seed in args.seeds for variant in args.variants]
    running, handles = [], []
    last_heartbeat = time.monotonic()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("Stopping JI-ADF jobs and their child processes")

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        while pending or running:
            used = {job["gpu"] for job in running}
            for gpu in range(torch.cuda.device_count()):
                if gpu in used or not pending:
                    continue
                variant, seed = pending.pop(0)
                job_dir = directory / variant / f"seed-{seed}"
                job_dir.mkdir(parents=True, exist_ok=True)
                log_path = job_dir / "train.log"
                handle = log_path.open("a")
                handles.append(handle)
                command = [sys.executable, "-u", str(Path(__file__).resolve().parents[1] / "scripts/ji_adf.py"), "train",
                           "--reference", str(reference), "--images", str(args.images), "--output", str(job_dir),
                           "--variant", variant, "--seed", str(seed), "--device", f"cuda:{gpu}", "--resume"]
                for key in TRAIN_KEYS:
                    command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
                if args.cache:
                    command += ["--cache", str(args.cache)]
                offset = log_path.stat().st_size
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                running.append({"process": process, "gpu": gpu, "name": f"{variant}/seed-{seed}", "log": log_path, "offset": offset})
                print(f"Starting {variant} seed {seed} on GPU {gpu}; log: {log_path}", flush=True)
            for job in list(running):
                code = job["process"].poll()
                with job["log"].open() as handle:
                    handle.seek(job["offset"])
                    fresh = handle.read()
                    job["offset"] = handle.tell()
                for line in fresh.splitlines():
                    print(f"{job['name']}: {line}", flush=True)
                if code is not None:
                    if code:
                        raise RuntimeError(f"Training failed: {job['name']}, exit={code}. Inspect {job['log']}")
                    running.remove(job)
                    print(f"Completed {job['name']}", flush=True)
            if running:
                if time.monotonic() - last_heartbeat >= 60:
                    print("Active jobs: " + ", ".join(job["name"] for job in running), flush=True)
                    last_heartbeat = time.monotonic()
                time.sleep(5)
    finally:
        for job in running:
            process = job["process"]
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for handle in handles:
            handle.close()
        signal.signal(signal.SIGTERM, previous_term)


def smoke(args):
    reference = Path(args.reference)
    verify_reference(reference)
    device = pick_device(args.device)
    torch.set_num_threads(2)
    encoder = MetadataEncoder().fit(paired_frame(reference / "split.csv").query("split == 'train'"))
    model = JIADFModel(encoder.input_dim_ + 2, variant=args.variant).to(device).to(memory_format=torch.channels_last)
    model.initialize(reference, args.initialization)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    images = [torch.randn(args.batch_size, 3, args.size, args.size, device=device).contiguous(memory_format=torch.channels_last)
              for _ in range(2)]
    metadata = torch.zeros(args.batch_size, encoder.input_dim_ + 2, device=device)
    targets = torch.arange(args.batch_size, device=device) % 11
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _step in range(3):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(args.accumulate):
            with _autocast(device, False):
                output = model(*images, metadata)
                loss = ji_adf_loss(output, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite preflight loss")
            scaler.scale(loss / args.accumulate).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.monotonic() - start)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    print(f"PASS {args.variant} full forward/backward: device={device}, {args.size}px, batch={args.batch_size}, "
          "precision=FP32, "
          f"accumulation={args.accumulate}, warm optimizer step={sum(timings[1:])/2:.3f}s, peak GiB={peak}. "
          "No trained checkpoint was saved.", flush=True)
