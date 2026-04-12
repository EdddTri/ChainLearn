#!/usr/bin/env python3
"""
launch_parallel.py -- Run experiment.py across all 8 RTX 3060 GPUs in parallel.

Each (dataset, noniid, seed) combo runs as its own subprocess on a dedicated GPU.
Each subprocess writes results to a private tmp CSV; this script merges them all
into the standard experiment_results.csv at the end (no concurrent-write conflicts).

Usage:
    cd federated-learning
    python launch_parallel.py                        # full run: 2 datasets × 3 noniid × 5 seeds
    python launch_parallel.py --seeds 2              # quick test (10 configs)
    python launch_parallel.py --gpus 4               # use only GPUs 0-3
    python launch_parallel.py --datasets pneumoniamnist --noniid mild moderate
    python launch_parallel.py --epochs 5             # fewer epochs for quick debugging

The script prints a live status line for each running job and a final cost estimate.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
SIM_DIR    = ROOT_DIR / "simulation"
RESULTS_DIR = SIM_DIR / "results"
SCRIPT     = SIM_DIR / "experiment.py"

# ── Default experiment grid ──────────────────────────────────────────────────
DEFAULT_DATASETS = ["pneumoniamnist", "dermamnist"]
DEFAULT_NONIIDS  = ["mild", "moderate", "severe"]
DEFAULT_SEEDS    = [42, 142, 242, 342, 442]   # matches experiment.py seed generation

HOURLY_RATE = 0.490  # $/hr for the 8× RTX 3060 machine


# ── Helpers ──────────────────────────────────────────────────────────────────
def merge_csvs(tmp_files: list[Path], output: Path) -> int:
    """Merge all per-process tmp CSVs into one master CSV. Returns total row count."""
    rows = []
    fieldnames = None
    for f in tmp_files:
        if not f.exists():
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(list(reader))

    if not rows:
        print("  [merge] No result rows found in tmp files.")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)

    # Load existing rows (if --resume was used before)
    existing_keys = set()
    if output.exists():
        with open(output, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing_keys.add((row["dataset"], row["noniid"], row["seed"], row["method"]))

    new_rows = [r for r in rows
                if (r["dataset"], r["noniid"], r["seed"], r["method"]) not in existing_keys]

    mode = "a" if output.exists() else "w"
    with open(output, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(new_rows)

    return len(new_rows)


# Expected method rows per (dataset, noniid, seed) — used to detect partial/crashed runs.
EXPECTED_METHODS = {
    "Centralized", "Local-Weak", "Local-Medium", "Local-Strong", "Local-Best",
    "FedAvg", "FedProx", "FedMD", "Ours", "EqualWt-Ens", "Ours-Dropout",
    "Abl:No CapMul", "Abl:No Conf", "Abl:No ECE", "Abl:No Bonus", "Abl:No PoC",
}


def poll_processes(procs: list[dict], failed: list[dict]) -> list[dict]:
    """Return list of still-running process dicts; append failures to `failed`."""
    for p in procs:
        if p["proc"].poll() is not None:
            p["done"] = True
            p["end_time"] = time.time()
            p["returncode"] = p["proc"].returncode
            elapsed = p["end_time"] - p["start_time"]
            status = "OK" if p["returncode"] == 0 else f"ERR({p['returncode']})"
            print(f"  [{status}] GPU {p['gpu']:d} | {p['dataset']:>16} | "
                  f"{p['noniid']:>8} | seed={p['seed']:4d} | {elapsed/60:.1f} min",
                  flush=True)
            if p["returncode"] != 0:
                failed.append(p)
    return [p for p in procs if not p.get("done")]


def tail_last_line(path: Path) -> str:
    """Return the last non-empty line of a log file (for live heartbeat)."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return "(no output yet)"
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Read last 2KB
            f.seek(max(0, size - 2048))
            tail = f.read().decode("utf-8", errors="replace")
        # tqdm uses \r; split on any newline-ish char and take last non-empty
        for line in reversed(tail.replace("\r", "\n").split("\n")):
            line = line.strip()
            if line:
                return line[:90]
        return "(no output yet)"
    except Exception:
        return "(log read error)"


def print_heartbeat(running: list[dict]) -> None:
    """Print one-line status per running job so user knows progress is live."""
    now = time.time()
    print(f"\n  -- heartbeat @ {time.strftime('%H:%M:%S')} | {len(running)} running --",
          flush=True)
    for p in running:
        elapsed_min = (now - p["start_time"]) / 60
        tail = tail_last_line(Path(p["log_path"]))
        print(f"    GPU {p['gpu']} | {p['dataset'][:12]:>12} | {p['noniid']:>8} | "
              f"seed={p['seed']:4d} | {elapsed_min:5.1f} min | {tail}",
              flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Parallel GPU launcher for experiment.py (8× RTX 3060)"
    )
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds to run (default: 5)")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs to use (auto-detects if omitted)")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=["pneumoniamnist", "dermamnist"],
                        help="Datasets to evaluate")
    parser.add_argument("--noniid", nargs="+", default=DEFAULT_NONIIDS,
                        choices=["mild", "moderate", "severe"],
                        help="Non-IID severity levels")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs per hospital (default: 15)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already present in experiment_results.csv")
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS[:args.seeds]

    # ── Detect GPUs ──────────────────────────────────────────────────────────
    num_gpus = args.gpus
    if num_gpus is None:
        try:
            import torch
            num_gpus = max(1, torch.cuda.device_count())
        except Exception:
            num_gpus = 1
    print(f"GPUs available: {num_gpus}  |  Parallelism: {num_gpus} concurrent experiments")

    # ── Build task list ───────────────────────────────────────────────────────
    all_tasks = list(product(args.datasets, args.noniid, seeds))

    # Filter already-completed if --resume: a combo counts as done only when
    # every EXPECTED_METHODS row is present. Partial rows from crashed jobs are rerun.
    skip = set()
    master_csv = RESULTS_DIR / "experiment_results.csv"
    if args.resume and master_csv.exists():
        combo_methods: dict = {}
        with open(master_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    key = (row["dataset"], row["noniid"], int(row["seed"]))
                except (ValueError, KeyError):
                    continue  # adversarial rows with non-int seeds
                combo_methods.setdefault(key, set()).add(row["method"])
        partial = 0
        for key, methods in combo_methods.items():
            if EXPECTED_METHODS.issubset(methods):
                skip.add(key)
            else:
                partial += 1
        print(f"Resuming: {len(skip)} combos complete; {partial} partial (will rerun)")

    tasks = [(ds, noniid, seed) for ds, noniid, seed in all_tasks
             if (ds, noniid, seed) not in skip]

    if not tasks:
        print("All tasks already complete. Nothing to run.")
        return

    print(f"Total configs to run: {len(tasks)}  |  Seeds: {seeds}")
    print(f"Grid: {args.datasets} × {args.noniid} × seeds")
    print(f"Estimated wall-clock (rough): {max(1, len(tasks) // num_gpus) * 10:.0f}–"
          f"{max(1, len(tasks) // num_gpus) * 15:.0f} min\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_csvs: list[Path] = []
    running: list[dict] = []
    failed: list[dict] = []
    t_start = time.time()
    HEARTBEAT_EVERY = 30  # seconds
    last_heartbeat = time.time()

    # Run adversarial experiments only for tasks whose seed matches compute_stats's
    # filter (adv_42). Every other subprocess skips adversarial to save time and avoid
    # emitting adv_142/242/... rows that the stats script ignores.
    ADVERSARIAL_SEED = DEFAULT_SEEDS[0]  # 42

    for task_idx, (dataset, noniid, seed) in enumerate(tasks):
        # Wait until a GPU is actually free (not just any slot)
        while True:
            running = poll_processes(running, failed)
            used_gpus = {p["gpu"] for p in running}
            free_gpus = [g for g in range(num_gpus) if g not in used_gpus]
            if free_gpus:
                gpu_id = free_gpus[0]
                break
            if time.time() - last_heartbeat > HEARTBEAT_EVERY and running:
                print_heartbeat(running)
                last_heartbeat = time.time()
            time.sleep(3)

        tmp_csv = RESULTS_DIR / f"_tmp_{task_idx:04d}_{dataset}_{noniid}_{seed}.csv"
        tmp_csvs.append(tmp_csv)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"  # ensure subprocess stdout is line-buffered -> heartbeat sees live tail
        # Reduces CUDA memory fragmentation — recommended by the OOM error itself
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        cmd = [
            sys.executable, str(SCRIPT),
            "--datasets", dataset,
            "--noniid",   noniid,
            "--seed-list", str(seed),
            "--epochs",   str(args.epochs),
            "--output-csv", str(tmp_csv),
        ]
        if seed != ADVERSARIAL_SEED:
            cmd.append("--skip-adversarial")

        log_path = RESULTS_DIR / f"_log_{task_idx:04d}_{dataset}_{noniid}_{seed}.txt"
        log_fh = open(log_path, "w", encoding="utf-8")

        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        running.append({
            "proc": proc,
            "gpu": gpu_id,
            "dataset": dataset,
            "noniid": noniid,
            "seed": seed,
            "start_time": time.time(),
            "log": log_fh,
            "log_path": str(log_path),
        })
        print(f"  [START] GPU {gpu_id} | {dataset:>16} | {noniid:>8} | seed={seed:4d}",
              flush=True)

    # Wait for all remaining jobs
    while running:
        time.sleep(3)
        running = poll_processes(running, failed)
        if time.time() - last_heartbeat > HEARTBEAT_EVERY and running:
            print_heartbeat(running)
            last_heartbeat = time.time()

    elapsed_total = time.time() - t_start

    # ── Merge tmp CSVs into master CSV ────────────────────────────────────────
    print(f"\nMerging {len(tmp_csvs)} result files into {master_csv} ...")
    total_rows = merge_csvs(tmp_csvs, master_csv)
    print(f"  Written {total_rows} result rows to {master_csv}")

    # Clean up tmp CSV files (logs are kept under simulation/results/_log_*.txt for debugging)
    for f in tmp_csvs:
        if f.exists():
            f.unlink()

    # ── Summary ───────────────────────────────────────────────────────────────
    cost = (elapsed_total / 3600) * HOURLY_RATE
    succeeded = len(tasks) - len(failed)
    status_label = "DONE" if not failed else "DONE WITH FAILURES"
    print(f"\n{'='*60}")
    print(f"  {status_label}  |  Wall-clock: {elapsed_total/60:.1f} min  |  Cost: ${cost:.3f}")
    print(f"  Configs: {succeeded} succeeded, {len(failed)} failed, {len(tasks)} total")
    if failed:
        print(f"  Failed jobs (rerun with --resume to retry):")
        for p in failed:
            print(f"    - GPU {p['gpu']} | {p['dataset']} | {p['noniid']} | "
                  f"seed={p['seed']} | rc={p.get('returncode')} | log={p['log_path']}")
    print(f"{'='*60}")
    if failed:
        sys.exit(1)
    print(f"\nNext steps:")
    print(f"  python simulation/compute_stats.py")
    print(f"  python simulation/generate_figures.py")


if __name__ == "__main__":
    main()
