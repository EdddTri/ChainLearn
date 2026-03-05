"""
contract_integration.py -- Web3.py <-> FLCoordinator bridge
==========================================================
Runs end-to-end FLCoordinator flow with real-data local training, signed PoC,
and multi-round submissions.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

sys.path.insert(0, os.path.dirname(__file__))
from capacity_manager import mock_training_manager

SCALE = 10_000
CAPACITY_CLASS_MAP = {"Weak": 0, "Medium": 1, "Strong": 2}

HARDHAT_ACCOUNTS = [
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
]

DEFAULT_ABI_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "smart_contracts",
    "artifacts",
    "contracts",
    "FLCoordinator.sol",
    "FLCoordinator.json",
)


class FLCoordinatorClient:
    """High-level Python client for the FLCoordinator contract."""

    def __init__(
        self,
        provider_url: str = "http://127.0.0.1:8545",
        contract_address: str = "",
        abi_path: str = DEFAULT_ABI_PATH,
        private_key: str = HARDHAT_ACCOUNTS[0],
    ):
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to {provider_url}")

        with open(abi_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        abi = artifact.get("abi", artifact)

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi,
        )

        self.account = self.w3.eth.account.from_key(private_key)
        self.address = self.account.address

    def _send_tx(self, fn) -> dict:
        tx = fn.build_transaction(
            {
                "from": self.address,
                "nonce": self.w3.eth.get_transaction_count(self.address),
                "gas": 3_000_000,
                "gasPrice": self.w3.eth.gas_price,
            }
        )
        signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

    @staticmethod
    def _hash_to_bytes32(hex_hash: str) -> bytes:
        return bytes.fromhex(hex_hash.removeprefix("0x")[:64])

    def register_hospital(
        self,
        hospital_address: str,
        name: str,
        capacity_class: str,
        benchmark_hash: bytes,
        benchmark_signature: bytes,
    ) -> dict:
        enum_val = CAPACITY_CLASS_MAP[capacity_class]
        return self._send_tx(
            self.contract.functions.registerHospital(
                Web3.to_checksum_address(hospital_address),
                name,
                enum_val,
                benchmark_hash,
                benchmark_signature,
            )
        )

    def start_round(self) -> int:
        self._send_tx(self.contract.functions.startNewRound())
        return self.contract.functions.currentRound().call()

    def submit_training_results(self, poc_results: Dict) -> dict:
        model_bytes = self._hash_to_bytes32(poc_results["model_hash"])
        confidence = int(poc_results["confidence"])
        ece = int(poc_results["ece"])
        return self._send_tx(
            self.contract.functions.submitUpdate(model_bytes, confidence, ece)
        )

    def get_all_weights(self, round_number: int) -> List[Dict]:
        submitter_count = self.contract.functions.getRoundSubmitterCount(round_number).call()
        if submitter_count == 0:
            return []

        capacity_labels = {0: "Weak", 1: "Medium", 2: "Strong"}
        rows = []
        for i in range(submitter_count):
            addr = self.contract.functions.roundSubmitters(round_number, i).call()
            info = self.contract.functions.getHospitalInfo(addr).call()
            weight = self.contract.functions.calculateWeight(addr).call()
            rows.append(
                {
                    "address": addr,
                    "name": info[0],
                    "capacity": capacity_labels.get(info[1], "Unknown"),
                    "confidence": info[2],
                    "ece": info[3],
                    "weight": weight,
                }
            )
        return rows

    def print_round_weights(self, round_number: int):
        rows = self.get_all_weights(round_number)
        if not rows:
            print(f"No submissions in round {round_number}")
            return

        print(f"\nRound {round_number} Weights")
        print(f"{'Hospital':<20} {'Class':<8} {'Conf':>7} {'ECE':>7} {'Weight':>8}")
        print(f"{'-' * 20} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 8}")
        for r in rows:
            print(
                f"{r['name']:<20} {r['capacity']:<8} {r['confidence']:>7} {r['ece']:>7} {r['weight']:>8}"
            )


def run_full_demo(contract_address: str, rounds: int = 3):
    owner_client = FLCoordinatorClient(
        contract_address=contract_address,
        private_key=HARDHAT_ACCOUNTS[0],
    )

    hospital_names = ["City General", "St. Marys", "University Med"]

    print("Phase 1: PoC + registration")
    registration = []
    for i, name in enumerate(hospital_names):
        poc = mock_training_manager(
            device="cpu",
            hospital_id=i,
            num_hospitals=3,
            run_poc=True,
            private_key=HARDHAT_ACCOUNTS[i + 1],
            epochs=1,
            max_samples=256,
        )
        hospital_account = Web3().eth.account.from_key(HARDHAT_ACCOUNTS[i + 1])
        owner_client.register_hospital(
            hospital_address=hospital_account.address,
            name=name,
            capacity_class=poc["capacity_class"],
            benchmark_hash=poc["benchmark_hash"],
            benchmark_signature=poc["benchmark_signature"],
        )
        registration.append(
            {
                "name": name,
                "capacity_class": poc["capacity_class"],
                "key": HARDHAT_ACCOUNTS[i + 1],
            }
        )
        print(
            f"Registered {name} as {poc['capacity_class']} throughput={poc['throughput']:.2f}"
        )

    print("\nPhase 2: multi-round submissions")
    for r in range(1, rounds + 1):
        started_round = owner_client.start_round()
        print(f"\nRound {started_round}")

        for i, row in enumerate(registration):
            node_client = FLCoordinatorClient(
                contract_address=contract_address,
                private_key=row["key"],
            )
            results = mock_training_manager(
                device="cpu",
                hospital_id=i,
                num_hospitals=3,
                run_poc=False,
                capacity_override=row["capacity_class"],
                epochs=1,
                max_samples=256,
            )
            node_client.submit_training_results(results)
            print(
                f"  {row['name']}: conf={results['confidence']}/{SCALE} ece={results['ece']}/{SCALE}"
            )

        owner_client.print_round_weights(started_round)


def query_round_weights(contract_address: str, round_number: int):
    client = FLCoordinatorClient(contract_address=contract_address)
    client.print_round_weights(round_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLCoordinator Web3.py integration")
    parser.add_argument(
        "--address",
        type=str,
        default=os.environ.get("CONTRACT_ADDRESS", ""),
        help="Deployed FLCoordinator contract address",
    )
    parser.add_argument(
        "--query",
        type=int,
        default=None,
        help="Query aggregation weights for a specific round",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of training rounds for demo mode",
    )
    args = parser.parse_args()

    if not args.address:
        print("No contract address provided. Set CONTRACT_ADDRESS or pass --address.")
        sys.exit(1)

    if args.query is not None:
        query_round_weights(args.address, args.query)
    else:
        run_full_demo(args.address, rounds=args.rounds)
