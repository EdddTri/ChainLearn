# ChainLearn: Blockchain-Based Capacity-Aware Federated Ensemble Learning

A federated learning framework for medical imaging that uses blockchain-based coordination, hardware-aware model assignment, and weighted ensemble aggregation. Hospital nodes undergo a Proof of Capacity (PoC) benchmark, receive capacity-appropriate model architectures, train locally on private data, and submit reliability metrics to an Ethereum smart contract. The final prediction is a weighted ensemble of heterogeneous models, with weights computed deterministically on-chain.

## Aim

Traditional federated learning assumes homogeneous compute across participants and aggregates parameters (e.g., FedAvg). This fails when hospitals have vastly different hardware capabilities, and parameter averaging across different architectures is impossible.

This project addresses both problems:

1. **Capacity-aware model assignment** -- A Proof of Capacity benchmark classifies each hospital node as Weak, Medium, or Strong, then assigns an architecture sized to its hardware (MobileNet, EfficientNet-B0, or ResNet-50).
2. **Ensemble prediction instead of parameter aggregation** -- Since each hospital trains a different architecture, predictions are combined via weighted softmax averaging rather than weight averaging.
3. **Deterministic weight formula** -- Aggregation weights are computed using a fixed-point formula based on capacity class, model confidence, calibration error (ECE), and participation history. The formula is implemented both in the Solidity smart contract (for on-chain transparency and auditability) and mirrored in Python (for experiment evaluation). Both implementations produce identical results.

## System Architecture

```
Hospital Node A          Hospital Node B          Hospital Node C
(Weak / MobileNet)       (Medium / EfficientNet)  (Strong / ResNet-50)
       |                        |                        |
       |-- PoC Benchmark -------|------------------------|
       |                        |                        |
       v                        v                        v
   Local Training           Local Training           Local Training
   (private data)           (private data)           (private data)
       |                        |                        |
       |-- Submit: model_hash, confidence, ECE, modelType
       |                        |                        |
       v                        v                        v
  +----------------------------------------------------------+
  |              FLCoordinator Smart Contract                 |
  |  - Verifies PoC signature (EIP-191 ECDSA)                |
  |  - Enforces model type matches capacity class             |
  |  - Computes aggregation weights on-chain                  |
  |  - Records ensemble prediction hash                       |
  +----------------------------------------------------------+
       |
       v
  Weighted Ensemble Prediction (softmax averaging)
```

## Workflow

### 1. Proof of Capacity (PoC) Benchmark
Each hospital runs a fixed K-step SGD benchmark on a small CNN to measure compute throughput (samples/sec):
- **Weak** (< 100 samples/sec) -- assigned **MobileNet-V3-Small**
- **Medium** (100-300 samples/sec) -- assigned **EfficientNet-B0**
- **Strong** (>= 300 samples/sec) -- assigned **ResNet-50**

The benchmark result is hashed (SHA-256) and signed with the hospital's private key (EIP-191).

### 2. Hospital Registration
The contract owner calls `registerHospital()` with the hospital's address, name, capacity class, benchmark hash, and ECDSA signature. The contract verifies the signature matches the hospital's address.

### 3. Training Rounds
Each round:
1. Owner calls `startNewRound()` to advance the round counter.
2. Each hospital trains its assigned model on its private local data shard.
3. Each hospital computes reliability metrics:
   - **Confidence**: mean max softmax probability over the evaluation set, scaled to [0, 10000].
   - **ECE** (Expected Calibration Error): equal-width binning over 10 bins, scaled to [0, 10000].
4. Each hospital calls `submitUpdate(modelHash, confidence, ece, modelType)`. The contract verifies the model type matches the hospital's capacity-assigned architecture.

### 4. On-Chain Weight Calculation
The weight formula (fixed-point arithmetic, SCALE = 10,000):

```
capMul = { Weak: 8000, Medium: 10000, Strong: 12000 }
baseWeight = capMul * confidence * (SCALE - ece) / SCALE^2
bonus = min(roundsParticipated * 500, 2500)
weight = min(baseWeight + bonus, 15000)
```

### 5. Weighted Ensemble Prediction
On the aggregator side, predictions from all models are combined:

```python
# For each model i with weight w_i:
output = sum(w_i * softmax(model_i(x))) / sum(w_i)
predicted_class = argmax(output)
```

The ensemble prediction hash is recorded on-chain via `recordEnsemblePrediction()` for auditability.

## Project Structure

```
ChainLearn/
|-- smart_contracts/
|   |-- contracts/
|   |   |-- FLCoordinator.sol       # Solidity smart contract
|   |-- test/
|   |   |-- FLCoordinator.test.js   # 59 Hardhat/Chai tests
|   |-- hardhat.config.js
|   |-- package.json
|
|-- hospital_node/
|   |-- capacity_manager.py     # PoC benchmark, model assignment, reliability metrics
|   |-- model.py                # Simple FederatedCNN (legacy/debug model)
|   |-- trainer.py              # Local training loop (train_one_epoch, evaluate)
|   |-- aggregator.py           # FedAvg implementation (baseline comparison)
|   |-- data_loader.py          # PneumoniaMNIST loader + synthetic data fallback
|   |-- blockchain_client.py    # Web3.py wrapper for FLCoordinator contract
|   |-- contract_integration.py # High-level demo: register + multi-round + weights
|   |-- config.json             # Node configuration template
|
|-- simulation/
|   |-- run_simulation.py       # End-to-end simulation with local Hardhat node
|   |-- experiment.py           # Comprehensive experiment: baselines, ablations, stats
|   |-- compute_stats.py        # Recomputes all cited figures from experiment_results.csv
|   |-- generate_figures.py     # Generates paper figures (simulation/figures/*.pdf)
|   |-- simulate_federation.py  # Basic FL simulation (synthetic data)
|   |-- test_blockchain_integration.py  # Integration smoke test (currently failing)
|   |-- figures/                # Generated PDFs per dataset:
|   |   |                       #   fig1_accuracy_{dataset}, fig2_ece_{dataset},
|   |   |                       #   fig3_ablation_{dataset}, fig4_communication
|   |-- results/
|   |   |-- experiment_results.csv   # Output from experiment.py (both datasets)
|   |   |-- computed_stats.txt       # Output from compute_stats.py
```

### File Descriptions

**`hospital_node/capacity_manager.py`** -- Core module. Runs the PoC benchmark (`proof_of_capacity()`), assigns model architecture by capacity class (`assign_model()`), computes ECE (`compute_ece()`), and provides `mock_training_manager()` which orchestrates the full pipeline: benchmark, model assignment, local training on a real data shard, and reliability metric computation.

**`hospital_node/blockchain_client.py`** -- Web3.py wrapper around the FLCoordinator contract. Provides typed methods for `register_hospital()`, `submit_update()`, `record_ensemble_prediction()`, `calculate_weight()`, `get_hospital_info()`, and `get_ensemble_record()`.

**`hospital_node/contract_integration.py`** -- Higher-level demo client (`FLCoordinatorClient`) that combines PoC, registration, multi-round training, and weight querying into a single `run_full_demo()` workflow.

**`hospital_node/trainer.py`** -- Standard PyTorch training loop with `train_one_epoch()`, `evaluate()`, and `run_training()`.

**`hospital_node/aggregator.py`** -- FedAvg implementation (`federated_average()`) used as a baseline comparison method.

**`hospital_node/data_loader.py`** -- Loads PneumoniaMNIST from local NPZ files, with an optional synthetic data fallback for debugging.

**`hospital_node/model.py`** -- Simple `FederatedCNN` used for basic experiments. The capacity-aware system uses MobileNet/EfficientNet/ResNet instead.

**`simulation/run_simulation.py`** -- End-to-end simulation that automatically starts a Hardhat node, deploys the contract, registers 3 hospitals with signed PoC, runs multi-round training, reads on-chain weights, computes weighted ensemble prediction, and records the ensemble hash on-chain.

**`simulation/experiment.py`** -- Comprehensive experiment framework supporting multiple datasets, non-IID severity levels, random seeds, baseline comparisons, ablation studies, adversarial experiments, and cost analysis. Outputs formatted tables and CSV results.

**`simulation/compute_stats.py`** -- Recomputes every number cited in the paper from `experiment_results.csv`. Loops over both datasets and prints mean ± std tables for main results, ablations, adversarial scenarios, communication cost, and gas cost. Saves full output to `simulation/results/computed_stats.txt`.

**`simulation/generate_figures.py`** -- Generates all paper figures as PDFs into `simulation/figures/`. Produces per-dataset figures for accuracy (fig1), ECE (fig2), and ablation (fig3), plus one shared communication cost figure (fig4).

## Smart Contract: FLCoordinator

Written in Solidity 0.8.24, deployed on a local Hardhat network.

### Key Components

| Component | Description |
|---|---|
| `CapacityClass` enum | `Weak`, `Medium`, `Strong` |
| `ModelType` enum | `Light` (MobileNet), `Medium` (EfficientNet), `Heavy` (ResNet) |
| `getModelType()` | Pure function mapping capacity class to model type |
| `registerHospital()` | Registers with ECDSA-verified PoC benchmark |
| `submitUpdate()` | 4-arg submission with model type enforcement |
| `calculateWeight()` / `calculateWeightPure()` | On-chain weight formula |
| `recordEnsemblePrediction()` | Records ensemble hash per round |
| `getHospitalInfo()` | Returns full hospital profile including assigned model type |
| `getEnsembleRecord()` | Returns ensemble hash, participant count, timestamp |

### Test Coverage

59 tests covering:
- Hospital registration and PoC signature verification
- Round lifecycle and submission validation
- Weight calculation across all capacity classes
- Participation bonus accumulation and cap
- Model type assignment and enforcement (wrong type rejection)
- Ensemble prediction recording (duplicates, zero participants, non-owner rejection)

## Experiment Framework

`simulation/experiment.py` runs a comprehensive evaluation with statistical rigor.

### Methods Compared

| Method | Description |
|---|---|
| **Centralized** | ResNet-50 trained on all pooled data (upper bound) |
| **Local-Best** | Best single-hospital model by accuracy |
| **Local-Weak/Medium/Strong** | Individual hospital models |
| **FedAvg** | Multi-round FL: all hospitals train ResNet-50, data-proportional parameter averaging, 5 rounds |
| **FedProx** | Multi-round FedAvg + proximal regularization term `(mu/2) * \|\|w - w_global\|\|^2`, 5 rounds |
| **FedMD** | Heterogeneous knowledge distillation via public dataset consensus logits |
| **EqualWt-Ens** | Same multi-round trained models as Ours, but with uniform weights (1/3 each) |
| **Ours** | 3 capacity-assigned models, multi-round training, on-chain capacity-aware weights with participation bonus |
| **Ours-Dropout** | Same as Ours but with realistic dropout (Weak=100%, Medium=80%, Strong=60% per-round attendance) |

### FedAvg / FedProx Baselines

Both FedAvg and FedProx run as proper multi-round federated learning (5 rounds). Each round: the global model is distributed to all hospitals, each hospital trains locally for `epochs // 5` epochs, then local models are aggregated using data-proportional weighting (`n_k / sum(n_k)`). The total compute budget (epochs) is the same across all methods for fair comparison.

FedProx adds a proximal term penalizing deviation from the global model (`mu = 0.01`):

```
loss = CrossEntropy(pred, label) + (mu / 2) * ||w_local - w_global||^2
```

### FedMD Baseline

FedMD (Federated Model Distillation) supports heterogeneous architectures like our method. It works by:
1. Splitting 10% of training data as a shared public dataset.
2. Each hospital trains its capacity-assigned model on its private shard.
3. All models compute softmax probabilities on the public dataset (temperature = 3.0); these are averaged into consensus probabilities.
4. Each model distills the consensus via KL divergence on the public set.
5. Final prediction is an equal-weight ensemble of the distilled models.

### Ablation Studies

| Ablation | Change from "Ours" |
|---|---|
| No CapMul | Capacity multiplier set to 1.0 for all |
| No Conf | Confidence set to 1.0 for all |
| No ECE | ECE set to 0 for all |
| No Bonus | Participation bonus disabled |
| No PoC | All hospitals train EfficientNet-B0, uniform capacity multiplier |

### Participation Bonus Analysis

The experiment includes a sensitivity analysis showing how the participation bonus interacts with capacity class across rounds. A weight-vs-rounds table is generated at fixed confidence/ECE, demonstrating:

- The participation bonus (capped at +2,500) partially compensates for lower capacity multipliers
- In the **dropout scenario** (Weak=100% attendance, Strong=60%), a reliable Weak hospital narrows the weight gap against an unreliable Strong hospital
- This validates the on-chain bonus mechanism as an incentive for consistent participation

### Adversarial Experiments

Three attack scenarios test the robustness of the on-chain coordination:

| Scenario | Description | Expected Outcome |
|---|---|---|
| **Lazy Hospital** | One hospital barely trains (1 epoch) and submits honest metrics | On-chain weights naturally downweight it due to low confidence and high ECE |
| **Inflated Metrics** | One hospital submits a bad model but lies about confidence (max) and ECE (zero) | Ensemble accuracy degrades -- motivates need for metric verification |
| **Capacity Spoofing** | A weak hospital claims strong capacity to get a heavier model | Without PoC: mismatch hurts ensemble. With PoC: the signed benchmark hash creates an auditable link between the hospital's identity and its claimed capacity, enabling detection of mismatches |

### Communication Cost Analysis

Per-round communication cost comparison across methods (upload + download per hospital):

| Method | Upload | Download | Total |
|---|---|---|---|
| **FedAvg / FedProx** | 102.2 MB | 102.2 MB | 204.5 MB |
| **FedMD** | 4.0 KB | 4.0 KB | 8.0 KB |
| **Ours** | 128 B | 96 B | **224 B** |

Ours transmits 4 × 32-byte ABI-encoded values (model hash, confidence, ECE, modelType) on upload and reads back 3 weights on download. This gives a **912,751× reduction** vs FedAvg, computed directly from parameter counts (25,557,032 params × 4 bytes = 102.2 MB per direction for ResNet-50).

### Gas Cost Analysis

Gas costs measured from the Hardhat test suite (`FLCoordinator.test.js`) at 20 Gwei, $2,000/ETH:

| Operation | Gas | USD | Frequency |
|---|---|---|---|
| `registerHospital()` | 174,764 | $6.99 | Once per hospital |
| `startNewRound()` | 48,942 | $1.96 | Once per round |
| `submitUpdate()` | 252,464 | $10.10 | Per hospital/round |
| `calculateWeight()` (view) | 0 | Free | Per hospital/round |
| `recordEnsemblePrediction()` | 94,931 | $3.80 | Once per round |
| **Total per round (3 hospitals)** | **901,265** | **$36.05** | |

`submitUpdate` dominates (3 × 252,464 = 757,392 gas) because it writes a full `Submission` struct to storage. L2 rollups reduce all costs by 10–100×.

### Datasets

Both datasets are sourced from [MedMNIST](https://medmnist.com/) and downloaded automatically on first run.

- **PneumoniaMNIST** -- Chest X-rays, 2 classes (normal/pneumonia). 4,708 train images, with the official test split (624 images) divided 50/50 into 312 validation / 312 test. Resized from 28×28 to 224×224×3 for ImageNet-compatible backbones.
- **DermaMNIST** -- Dermatoscopy images, 7 classes. Resized from 28×28 to 224×224×3.

In both cases, the validation split is used exclusively for computing reliability metrics (confidence, ECE) that feed into aggregation weight calculation. The test split is held out and used only for final evaluation.

### Non-IID Data Splitting

Data is split across 3 hospitals using a Dirichlet distribution to simulate label heterogeneity:

| Severity | Dirichlet alpha | Description |
|---|---|---|
| Mild | 1.0 | Near-uniform class distribution |
| Moderate | 0.5 | Moderate label skew |
| Severe | 0.1 | Extreme label imbalance |

### Metrics

- **Accuracy** -- Standard classification accuracy
- **F1-Score** -- Macro-averaged F1
- **ECE** -- Expected Calibration Error

All reported as mean +/- std across multiple random seeds.

### Experiment CLI Usage

```bash
# Full run (5 seeds, both datasets, all non-IID levels)
python simulation/experiment.py

# Quick run (1 seed, 1 epoch)
python simulation/experiment.py --seeds 1 --epochs 1

# Single dataset
python simulation/experiment.py --datasets pneumoniamnist

# Specific non-IID levels
python simulation/experiment.py --noniid moderate severe

# Results saved to simulation/results/experiment_results.csv
```

## Quick Start

### 1. Smart Contracts

```bash
cd smart_contracts
npm install
npx hardhat compile
npx hardhat test          # 59 tests
```

### 2. Hospital Node (Python)

```bash
cd hospital_node
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch torchvision web3 eth-account medmnist scikit-learn numpy tqdm python-dotenv
```

### 3. End-to-End Simulation

Starts Hardhat, deploys contract, runs PoC + training + ensemble automatically:

```bash
python simulation/run_simulation.py              # 3 rounds (default)
python simulation/run_simulation.py --rounds 5   # 5 rounds
```

### 4. Run Experiments

```bash
python simulation/experiment.py --seeds 3 --epochs 5
```

### 5. Reproduce Paper Numbers and Figures

```bash
# Verify all cited statistics match experiment_results.csv
# Output also saved to simulation/results/computed_stats.txt
python simulation/compute_stats.py

# Regenerate all paper figures into simulation/figures/
pip install matplotlib
python simulation/generate_figures.py
```

## Prerequisites

### Python
- Python 3.10+
- PyTorch, torchvision
- web3.py, eth-account, python-dotenv
- medmnist, scikit-learn
- numpy, tqdm
- matplotlib (for figure generation)

### Node.js
- Node.js 18+
- Hardhat, ethers.js
- OpenZeppelin Contracts

## Key Design Decisions

1. **Ensemble over FedAvg** -- Heterogeneous architectures cannot be parameter-averaged. Softmax ensemble preserves the strengths of each architecture.
2. **Fixed-point arithmetic** -- Solidity lacks floating-point. All percentages use SCALE = 10,000 (100% = 10000).
3. **Weight cap at 15,000** -- Prevents any single hospital from dominating the ensemble (max 1.5x multiplier).
4. **EIP-191 signatures** -- PoC benchmark results are signed by the hospital's private key and verified on-chain. The contract confirms *who* signed the benchmark hash, but the capacity class is a separate parameter set by the contract owner at registration. This means PoC provides an auditable identity link, not automatic capacity verification.
5. **Model type enforcement** -- The contract rejects submissions where the model type doesn't match the hospital's assigned architecture, ensuring the capacity-aware design is respected.
6. **Minimal communication** -- Only hashes and scalar metrics are sent on-chain (~224 bytes per hospital per round), compared to ~94 MB for FedAvg parameter sharing.
7. **Quality-aware weighting** -- The weight formula naturally downweights low-confidence or poorly calibrated models when metrics are reported honestly. This provides a passive defense but does not verify metric truthfulness; adversarial experiments quantify the impact of dishonest reporting.

## Limitations

1. **Metrics are self-reported** -- Confidence and ECE are submitted by each hospital with no verification. Our adversarial experiments quantify the impact of dishonest reporting.
2. **No data poisoning defense** -- A hospital can train on corrupted data while reporting honest metrics. Backdoor attacks that perform well on clean evaluation data are not detected.
3. **Benchmark replay** -- A hospital could run PoC on rented hardware, obtain a "Strong" signature, then train on weaker hardware. The signed hash is valid but may not reflect current capability.
4. **Experiments use Python-mirrored weights** -- The experiment pipeline computes weights using a Python function (`compute_weight`) that mirrors the Solidity `calculateWeightPure` formula exactly. Gas costs are estimated from Hardhat tests, not from actual on-chain experiment execution.

## Future Work

1. Decentralized metric attestation (e.g., peer or committee-based validation with dispute resolution)
2. Robust aggregation and poisoning defenses (Byzantine-resilient scoring/filtering)
3. Periodic or challenge-based re-benchmarking to prevent stale capacity claims
4. Full on-chain execution benchmarks on a live network for end-to-end gas validation

## License

MIT
