"""
compute_stats.py
========================
Recomputes every number cited in the paper from experiment_results.csv
and the constants defined in experiment.py.

Run:
    python compute_stats.py
"""

import csv
import statistics
from pathlib import Path

CSV_PATH = Path(__file__).parent / "simulation/results/experiment_results.csv"

# ── Constants (must match experiment.py and FLCoordinator.sol) ──────────────
NUM_HOSPITALS   = 3
NUM_CLASSES     = 2
SCALE           = 10_000

# Model parameter counts (float32, 4 bytes each)
RESNET50_PARAMS      = 25_557_032   # Strong
EFFICIENTNETB0_PARAMS = 5_288_548   # Medium  (for FedMD public-data calc)
BYTES_PER_PARAM      = 4

# FedMD public dataset size used in experiment.py
FEDMD_PUBLIC_SAMPLES = 500

# Our on-chain payload (bytes)
OURS_UPLOAD   = 128   # 4 x 32-byte ABI-encoded values: hash, conf, ece, modelType
OURS_DOWNLOAD = NUM_HOSPITALS * 32   # read back N weights

# Gas costs measured from Hardhat test suite (FLCoordinator.test.js)
GAS_COSTS = {
    "registerHospital":          174_764,
    "startNewRound":              48_942,
    "submitUpdate":              252_464,
    "calculateWeight (view)":          0,
    "recordEnsemblePrediction":   94_931,
}
GAS_PRICE_GWEI = 20
ETH_USD        = 2_000


# ── Load CSV ────────────────────────────────────────────────────────────────
def load_rows():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


# ── Statistics helpers ───────────────────────────────────────────────────────
def mean_std(values):
    if not values:
        return None, None
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def collect(rows, method, noniid, dataset="pneumoniamnist",
            seeds=("42", "142", "242", "342", "442")):
    accs, f1s, eces = [], [], []
    for r in rows:
        if (r["method"] == method and r["noniid"] == noniid
                and r["dataset"] == dataset and r["seed"] in seeds):
            accs.append(float(r["accuracy"]))
            f1s.append(float(r["f1_score"]))
            eces.append(float(r["ece"]))
    return accs, f1s, eces


# ── Section 1: Main comparison table ────────────────────────────────────────
def print_main_results(rows):
    main_methods = [
        "Centralized", "Local-Best", "FedAvg", "FedProx",
        "FedMD", "EqualWt-Ens", "Ours",
    ]
    print("\n" + "=" * 80)
    print("  MAIN COMPARISON — PneumoniaMNIST (mean ± std, 5 seeds)")
    print("=" * 80)

    for noniid, alpha in [("mild", "1.0"), ("moderate", "0.5"), ("severe", "0.1")]:
        print(f"\n  Non-IID: {noniid} (alpha={alpha})")
        print(f"  {'Method':<22} {'Accuracy':>18} {'Macro-F1':>18} {'ECE':>18}")
        print(f"  {'-'*22} {'-'*18} {'-'*18} {'-'*18}")
        for m in main_methods:
            accs, f1s, eces = collect(rows, m, noniid)
            if not accs:
                continue
            am, as_ = mean_std(accs)
            fm, fs  = mean_std(f1s)
            em, es  = mean_std(eces)
            print(f"  {m:<22} {am:.4f} ± {as_:.4f}   {fm:.4f} ± {fs:.4f}   {em:.4f} ± {es:.4f}")


# ── Section 2: Ablation table ────────────────────────────────────────────────
def print_ablation_results(rows):
    ablations = [
        "Ours", "Abl:No CapMul", "Abl:No Conf",
        "Abl:No ECE", "Abl:No Bonus", "Abl:No PoC",
    ]
    print("\n" + "=" * 80)
    print("  ABLATION STUDY — PneumoniaMNIST (mean ± std, 5 seeds)")
    print("=" * 80)

    for noniid, alpha in [("mild", "1.0"), ("moderate", "0.5"), ("severe", "0.1")]:
        print(f"\n  Non-IID: {noniid} (alpha={alpha})")
        print(f"  {'Method':<22} {'Accuracy':>18} {'Macro-F1':>18} {'ECE':>18}")
        print(f"  {'-'*22} {'-'*18} {'-'*18} {'-'*18}")
        for m in ablations:
            accs, f1s, eces = collect(rows, m, noniid)
            if not accs:
                continue
            am, as_ = mean_std(accs)
            fm, fs  = mean_std(f1s)
            em, es  = mean_std(eces)
            print(f"  {m:<22} {am:.4f} ± {as_:.4f}   {fm:.4f} ± {fs:.4f}   {em:.4f} ± {es:.4f}")


# ── Section 3: Adversarial results ──────────────────────────────────────────
def print_adversarial_results(rows):
    adv_methods = [
        "No attack (Ours)", "No attack (EqualWt)",
        "Lazy (Ours)", "Lazy (EqualWt)",
        "Inflated (Ours)", "Inflated (EqualWt)",
        "Spoofed (no PoC)", "Spoofed (with PoC)", "Spoofed (EqualWt, no PoC)",
    ]
    print("\n" + "=" * 80)
    print("  ADVERSARIAL ROBUSTNESS — seed=adv_42")
    print("=" * 80)

    for noniid in ["mild", "moderate", "severe"]:
        print(f"\n  Non-IID: {noniid}")
        print(f"  {'Scenario':<35} {'Accuracy':>10} {'Macro-F1':>10} {'ECE':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
        for m in adv_methods:
            for r in rows:
                if r["method"] == m and r["noniid"] == noniid and r["seed"] == "adv_42":
                    print(f"  {m:<35} {float(r['accuracy']):.4f}     "
                          f"{float(r['f1_score']):.4f}     {float(r['ece']):.4f}")


# ── Section 4: Communication cost ───────────────────────────────────────────
def print_communication_cost():
    def fmt(n):
        if n < 1000:
            return f"{n} B"
        elif n < 1_000_000:
            return f"{n / 1000:.1f} KB"
        else:
            return f"{n / 1_000_000:.1f} MB"

    fa_up   = RESNET50_PARAMS * BYTES_PER_PARAM
    fa_down = fa_up
    fa_tot  = fa_up + fa_down

    fedmd_up   = FEDMD_PUBLIC_SAMPLES * NUM_CLASSES * BYTES_PER_PARAM
    fedmd_down = fedmd_up
    fedmd_tot  = fedmd_up + fedmd_down

    ours_tot = OURS_UPLOAD + OURS_DOWNLOAD
    ratio    = fa_tot / ours_tot

    print("\n" + "=" * 80)
    print("  COMMUNICATION COST — per hospital per round")
    print("=" * 80)
    print(f"  {'Method':<25} {'Upload':>12} {'Download':>12} {'Total':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'FedAvg / FedProx':<25} {fmt(fa_up):>12} {fmt(fa_down):>12} {fmt(fa_tot):>12}")
    print(f"  {'FedMD':<25} {fmt(fedmd_up):>12} {fmt(fedmd_down):>12} {fmt(fedmd_tot):>12}")
    print(f"  {'Ours':<25} {fmt(OURS_UPLOAD):>12} {fmt(OURS_DOWNLOAD):>12} {fmt(ours_tot):>12}")
    print(f"\n  Reduction vs FedAvg: {ratio:,.0f}x")
    print(f"  ResNet-50 params: {RESNET50_PARAMS:,}  x4 bytes = {fmt(fa_up)} per direction")
    print(f"  Ours upload breakdown: hash=32B + conf=32B + ece=32B + modelType=32B = {OURS_UPLOAD}B")
    print(f"  Ours download: {NUM_HOSPITALS} weights x 32B = {OURS_DOWNLOAD}B")


# ── Section 5: Gas cost ──────────────────────────────────────────────────────
def print_gas_cost():
    freq_map = {
        "registerHospital":          "Once per hospital",
        "startNewRound":             "Once per round",
        "submitUpdate":              "Per hospital/round",
        "calculateWeight (view)":    "Per hospital/round",
        "recordEnsemblePrediction":  "Once per round",
    }

    total_per_round = (
        GAS_COSTS["startNewRound"]
        + NUM_HOSPITALS * GAS_COSTS["submitUpdate"]
        + GAS_COSTS["recordEnsemblePrediction"]
    )
    total_eth = total_per_round * GAS_PRICE_GWEI / 1e9
    total_usd = total_eth * ETH_USD

    print("\n" + "=" * 80)
    print(f"  GAS COST ANALYSIS (Hardhat-measured, {GAS_PRICE_GWEI} Gwei, ${ETH_USD}/ETH)")
    print("=" * 80)
    print(f"  {'Operation':<35} {'Gas':>10} {'USD':>10}  Frequency")
    print(f"  {'-'*35} {'-'*10} {'-'*10}  {'-'*20}")
    for op, gas in GAS_COSTS.items():
        usd = gas * GAS_PRICE_GWEI / 1e9 * ETH_USD
        cost_str = f"${usd:.4f}" if gas > 0 else "Free"
        print(f"  {op:<35} {gas:>10,} {cost_str:>10}  {freq_map[op]}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    print(f"  {'Total per round (3 hospitals)':<35} {total_per_round:>10,} ${total_usd:.4f}")
    print(f"\n  Note: submitUpdate dominates ({NUM_HOSPITALS} x {GAS_COSTS['submitUpdate']:,} = "
          f"{NUM_HOSPITALS * GAS_COSTS['submitUpdate']:,} gas) because it writes a Submission struct.")
    print(f"  L2 rollups reduce costs by 10-100x.")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}\nRun experiment.py first.")

    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {CSV_PATH.name}")
    print(f"Methods : {sorted(set(r['method'] for r in rows))}")
    print(f"NonIID  : {sorted(set(r['noniid'] for r in rows))}")
    print(f"Seeds   : {sorted(set(r['seed'] for r in rows))}")

    print_main_results(rows)
    print_ablation_results(rows)
    print_adversarial_results(rows)
    print_communication_cost()
    print_gas_cost()
    print()