# Decentralized Federated Learning

A monorepo for blockchain-orchestrated federated learning across hospital nodes.

## Project Structure

```text
federated-learning/
|-- smart_contracts/        # Solidity + Hardhat
|   |-- contracts/          # Solidity smart contracts
|   |-- scripts/            # Deploy scripts
|   |-- test/               # Contract tests (Chai + Ethers.js)
|   |-- hardhat.config.js
|   `-- package.json
|
|-- hospital_node/          # Python + PyTorch + Web3.py
|   |-- model.py            # CNN architecture
|   |-- trainer.py          # Local training loop
|   |-- aggregator.py       # FedAvg aggregation
|   |-- blockchain_client.py# Web3 contract interface
|   |-- data_loader.py      # Data utilities
|   |-- config.json         # Node configuration
|   `-- requirements.txt
|
|-- simulation/             # Local testing scripts
|   |-- requirements.txt    # Simulation/experiment dependencies
|   |-- simulate_federation.py          # Full FL simulation
|   `-- test_blockchain_integration.py  # Contract integration test
|
|-- .env.example            # Environment variable template
`-- README.md
```

## Quick Start

### 1. Smart Contracts (Node.js)

```bash
cd smart_contracts
npm install
npx hardhat compile
npx hardhat test
```

### 2. Hospital Node (Python)

```bash
cd hospital_node
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
# from repository root
cp .env.example .env
# Fill .env with local keys and config values
```

Required keys for simulation/integration scripts:
- `FL_PRIVATE_KEY`
- `FL_OWNER_PRIVATE_KEY`
- `FL_HOSPITAL1_PRIVATE_KEY`
- `FL_HOSPITAL2_PRIVATE_KEY`
- `FL_HOSPITAL3_PRIVATE_KEY`

### 4. Simulation / Experiment Dependencies

```bash
cd simulation
pip install -r requirements.txt
```

### 5. Run Simulation

```bash
cd simulation
python simulate_federation.py --nodes 3 --rounds 5 --epochs 2
```

### 6. Blockchain Integration Test

```bash
# Terminal 1 - start local blockchain
cd smart_contracts
npx hardhat node

# Terminal 2 - deploy contract
cd smart_contracts
npx hardhat run scripts/deploy_coordinator.js --network localhost
# Copy the deployed address

# Terminal 3 - run integration test
cd simulation
set CONTRACT_ADDRESS=0x...   # paste address
python test_blockchain_integration.py
```

## Architecture

```text
Hospital A --\                    /-- Hospital A
Hospital B ----> Smart Contract ---- Hospital B
Hospital C --/    (Ethereum)     \-- Hospital C
     |                                    |
     `-- Local Training <-- FedAvg -------`
```

Each hospital trains a CNN locally on private patient data.
Model weight hashes are submitted to the blockchain for
transparency and auditability. The FedAvg aggregation
produces a new global model each round.

## License

MIT
