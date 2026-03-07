"""
run_simulation.py -- End-to-end FLCoordinator simulation
========================================================
1. Starts local Hardhat node.
2. Deploys FLCoordinator.
3. Registers hospitals with signed PoC benchmark hashes.
4. Runs multi-round local training on non-IID real-data shards.
5. Submits model hash + reliability each round.
6. Reads on-chain weights and computes weighted ensemble output.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import List

import torch
import torch.nn.functional as F
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMART_CONTRACTS = os.path.join(ROOT_DIR, "smart_contracts")
ABI_PATH = os.path.join(
    SMART_CONTRACTS,
    "artifacts",
    "contracts",
    "FLCoordinator.sol",
    "FLCoordinator.json",
)

sys.path.insert(0, os.path.join(ROOT_DIR, "hospital_node"))
from capacity_manager import mock_training_manager

ACCOUNTS = [
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
]

CAPACITY_ENUM = {"Weak": 0, "Medium": 1, "Strong": 2}
MODEL_TYPE_ENUM = {"Weak": 0, "Medium": 1, "Strong": 2}  # Light=0, Medium=1, Heavy=2
SCALE = 10_000
NUM_CLASSES = 2
PROVIDER = "http://127.0.0.1:8545"


class ContractHelper:
    def __init__(self, contract_address: str, private_key: str):
        self.w3 = Web3(Web3.HTTPProvider(PROVIDER))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        assert self.w3.is_connected(), "Hardhat node not reachable"

        with open(ABI_PATH, "r", encoding="utf-8") as f:
            abi = json.load(f)["abi"]

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi,
        )
        self.acct = self.w3.eth.account.from_key(private_key)
        self.addr = self.acct.address

    def _tx(self, fn):
        tx = fn.build_transaction(
            {
                "from": self.addr,
                "nonce": self.w3.eth.get_transaction_count(self.addr),
                "gas": 5_000_000,
                "gasPrice": self.w3.eth.gas_price,
            }
        )
        signed = self.w3.eth.account.sign_transaction(tx, self.acct.key)
        return self.w3.eth.wait_for_transaction_receipt(
            self.w3.eth.send_raw_transaction(signed.raw_transaction)
        )


def start_hardhat() -> subprocess.Popen:
    print("[1/6] Starting Hardhat node...")
    env = os.environ.copy()
    env["HARDHAT_TELEMETRY_OPTOUT"] = "true"
    proc = subprocess.Popen(
        "npx hardhat node",
        cwd=SMART_CONTRACTS,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        shell=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    w3 = Web3(Web3.HTTPProvider(PROVIDER))
    for _ in range(30):
        time.sleep(1)
        if w3.is_connected():
            time.sleep(3)
            print("       Hardhat node is UP")
            return proc

    proc.kill()
    raise RuntimeError("Hardhat node failed to start")


def deploy_contract() -> str:
    print("[2/6] Deploying FLCoordinator...")
    w3 = Web3(Web3.HTTPProvider(PROVIDER))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    with open(ABI_PATH, "r", encoding="utf-8") as f:
        artifact = json.load(f)

    acct = w3.eth.account.from_key(ACCOUNTS[0])
    contract = w3.eth.contract(abi=artifact["abi"], bytecode=artifact["bytecode"])

    tx = contract.constructor().build_transaction(
        {
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": 5_000_000,
            "gasPrice": w3.eth.gas_price,
        }
    )
    signed = w3.eth.account.sign_transaction(tx, acct.key)
    receipt = w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(signed.raw_transaction))
    print(f"       Contract deployed at {receipt.contractAddress}")
    return receipt.contractAddress


def weighted_ensemble_predict(models_list, weights: List[float], x: torch.Tensor) -> torch.Tensor:
    total_w = sum(weights)
    normed = [w / total_w for w in weights]

    out = torch.zeros(x.size(0), NUM_CLASSES)
    for model, w in zip(models_list, normed):
        model.eval()
        with torch.no_grad():
            out += w * F.softmax(model(x), dim=1).cpu()
    return out


def main(rounds: int):
    hardhat_proc = None
    try:
        hardhat_proc = start_hardhat()
        contract_addr = deploy_contract()
        owner = ContractHelper(contract_addr, ACCOUNTS[0])

        print("[3/6] Registering hospitals with signed PoC...")
        profiles = []
        for i, key in enumerate(ACCOUNTS[1:4]):
            name = f"Hospital-{chr(ord('A') + i)}"
            reg = mock_training_manager(
                device="cpu",
                hospital_id=i,
                num_hospitals=3,
                run_poc=True,
                private_key=key,
                epochs=1,
                max_samples=256,
            )
            addr = Web3().eth.account.from_key(key).address
            owner._tx(
                owner.contract.functions.registerHospital(
                    Web3.to_checksum_address(addr),
                    name,
                    CAPACITY_ENUM[reg["capacity_class"]],
                    reg["benchmark_hash"],
                    reg["benchmark_signature"],
                )
            )
            profiles.append(
                {
                    "name": name,
                    "key": key,
                    "capacity_class": reg["capacity_class"],
                }
            )
            print(
                f"       {name}: class={reg['capacity_class']} throughput={reg['throughput']:.2f}"
            )

        last_models = None
        last_weights = None

        print("[4/6] Running multi-round training...")
        for r in range(1, rounds + 1):
            owner._tx(owner.contract.functions.startNewRound())
            current_round = owner.contract.functions.currentRound().call()
            print(f"\n       Round {current_round}")

            trained_models = []
            for i, hp in enumerate(profiles):
                node_client = ContractHelper(contract_addr, hp["key"])
                results = mock_training_manager(
                    device="cpu",
                    hospital_id=i,
                    num_hospitals=3,
                    run_poc=False,
                    capacity_override=hp["capacity_class"],
                    epochs=1,
                    max_samples=256,
                )
                model_hash = bytes.fromhex(results["model_hash"].removeprefix("0x")[:64])
                model_type = MODEL_TYPE_ENUM[hp["capacity_class"]]
                node_client._tx(
                    node_client.contract.functions.submitUpdate(
                        model_hash,
                        int(results["confidence"]),
                        int(results["ece"]),
                        model_type,
                    )
                )
                trained_models.append(results["model"])
                print(
                    f"         {hp['name']}: conf={results['confidence']}/{SCALE} ece={results['ece']}/{SCALE}"
                )

            print("[5/6] On-chain weights")
            on_chain_weights = []
            for hp in profiles:
                h_addr = Web3().eth.account.from_key(hp["key"]).address
                w = owner.contract.functions.calculateWeight(Web3.to_checksum_address(h_addr)).call()
                on_chain_weights.append(w)
                print(f"         {hp['name']}: w={w}")

            last_models = trained_models
            last_weights = on_chain_weights

        print("[6/6] Weighted ensemble prediction")
        dummy_input = torch.randn(1, 3, 224, 224)
        y_final = weighted_ensemble_predict(last_models, [float(w) for w in last_weights], dummy_input)
        pred = y_final.argmax(dim=1).item()
        conf = y_final.max(dim=1).values.item()
        print(f"       Predicted class: {pred}")
        print(f"       Confidence: {conf:.4f}")

        # Record ensemble prediction hash on-chain
        import hashlib
        pred_bytes = y_final.numpy().tobytes()
        pred_hash = hashlib.sha256(pred_bytes).digest()
        owner._tx(
            owner.contract.functions.recordEnsemblePrediction(pred_hash, len(profiles))
        )
        print(f"       Ensemble hash recorded on-chain: 0x{pred_hash.hex()[:16]}...")

    finally:
        if hardhat_proc and hardhat_proc.poll() is None:
            print("\nShutting down Hardhat node...")
            if sys.platform == "win32":
                hardhat_proc.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(2)
                if hardhat_proc.poll() is None:
                    hardhat_proc.kill()
            else:
                hardhat_proc.terminate()
                hardhat_proc.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLCoordinator simulation")
    parser.add_argument("--rounds", type=int, default=3, help="Number of training rounds")
    args = parser.parse_args()
    main(rounds=args.rounds)
