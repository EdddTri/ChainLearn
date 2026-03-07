# Capacity-Aware Decentralized Federated Ensemble Learning with Blockchain Coordination

A federated learning framework for medical imaging that uses blockchain-based coordination, hardware-aware model assignment, and weighted ensemble aggregation. Hospital nodes undergo a Proof of Capacity (PoC) benchmark, receive capacity-appropriate model architectures, train locally on private data, and submit reliability metrics to an Ethereum smart contract. The final prediction is a weighted ensemble of heterogeneous models, with weights computed deterministically on-chain.

## Aim

Traditional federated learning assumes homogeneous compute across participants and aggregates parameters (e.g., FedAvg). This fails when hospitals have vastly different hardware capabilities, and parameter averaging across different architectures is impossible.

This project addresses both problems:

1. **Capacity-aware model assignment** -- A Proof of Capacity benchmark classifies each hospital node as Weak, Medium, or Strong, then assigns an architecture sized to its hardware (MobileNet, EfficientNet-B0, or ResNet-50).
2. **Ensemble prediction instead of parameter aggregation** -- Since each hospital trains a different architecture, predictions are combined via weighted softmax averaging rather than weight averaging.
3. **On-chain weight computation** -- Aggregation weights are computed deterministically on an Ethereum smart contract using capacity class, model confidence, calibration error (ECE), and participation history. This ensures transparency and auditability.

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
federated-learning/
|-- smart_contracts/
|   |-- contracts/
|   |   |-- FLCoordinator.sol       # Solidity smart contract
|   |-- test/
|   |   |-- FLCoordinator.test.js   # 47 Hardhat/Chai tests
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
|   |-- simulate_federation.py  # Basic FL simulation (synthetic data)
|   |-- test_blockchain_integration.py  # Integration smoke test
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

**`simulation/experiment.py`** -- Comprehensive experiment framework supporting multiple datasets, non-IID severity levels, random seeds, baseline comparisons, and ablation studies. Outputs formatted tables and CSV results.

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

47 tests covering:
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
| **FedAvg** | All hospitals train ResNet-50, average parameters |
| **EqualWt-Ens** | 3 capacity-assigned models, uniform weights (1/3 each) |
| **Ours** | 3 capacity-assigned models, on-chain capacity-aware weights |

### Ablation Studies

| Ablation | Change from "Ours" |
|---|---|
| No CapMul | Capacity multiplier set to 1.0 for all |
| No Conf | Confidence set to 1.0 for all |
| No ECE | ECE set to 0 for all |
| No Bonus | Participation bonus disabled |
| No PoC | All hospitals train EfficientNet-B0, uniform capacity multiplier |

### Datasets

- **PneumoniaMNIST** -- Chest X-rays, 2 classes (normal/pneumonia)
- **DermaMNIST** -- Dermatoscopy images, 7 classes

Both sourced from [MedMNIST](https://medmnist.com/).

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
npx hardhat test          # 47 tests
```

### 2. Hospital Node (Python)

```bash
cd hospital_node
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch torchvision web3 eth-account medmnist scikit-learn numpy
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

## Prerequisites

### Python
- Python 3.10+
- PyTorch, torchvision
- web3.py, eth-account
- medmnist, scikit-learn
- numpy

### Node.js
- Node.js 18+
- Hardhat, ethers.js
- OpenZeppelin Contracts

## Key Design Decisions

1. **Ensemble over FedAvg** -- Heterogeneous architectures cannot be parameter-averaged. Softmax ensemble preserves the strengths of each architecture.
2. **Fixed-point arithmetic** -- Solidity lacks floating-point. All percentages use SCALE = 10,000 (100% = 10000).
3. **Weight cap at 15,000** -- Prevents any single hospital from dominating the ensemble (max 1.5x multiplier).
4. **EIP-191 signatures** -- PoC benchmark results are signed by the hospital's private key and verified on-chain, preventing spoofed capacity claims.
5. **Model type enforcement** -- The contract rejects submissions where the model type doesn't match the hospital's assigned architecture, ensuring the capacity-aware design is respected.

## License

MIT
