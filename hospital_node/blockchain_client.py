"""
blockchain_client.py -- Web3 interface for FLCoordinator
========================================================
"""

import json
import os

from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


class BlockchainClient:
    """Client for FLCoordinator contract interactions."""

    def __init__(
        self,
        provider_url: str,
        contract_address: str,
        abi_path: str,
        private_key: str | None = None,
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

        key = private_key or _require_env("FL_PRIVATE_KEY")
        self.account = self.w3.eth.account.from_key(key)
        self.address = self.account.address

    def _send_tx(self, fn):
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

    def register_hospital(
        self,
        hospital_address: str,
        name: str,
        capacity_class: int,
        benchmark_hash: bytes,
        benchmark_signature: bytes,
    ) -> dict:
        return self._send_tx(
            self.contract.functions.registerHospital(
                Web3.to_checksum_address(hospital_address),
                name,
                capacity_class,
                benchmark_hash,
                benchmark_signature,
            )
        )

    def start_new_round(self) -> dict:
        return self._send_tx(self.contract.functions.startNewRound())

    def submit_update(self, model_hash: bytes, confidence: int, ece: int) -> dict:
        return self._send_tx(
            self.contract.functions.submitUpdate(model_hash, confidence, ece)
        )

    def get_current_round(self) -> int:
        return self.contract.functions.currentRound().call()

    def get_hospital_count(self) -> int:
        return self.contract.functions.getHospitalCount().call()

    def get_round_submission(self, round_number: int, hospital_address: str):
        return self.contract.functions.getRoundSubmission(
            round_number,
            Web3.to_checksum_address(hospital_address),
        ).call()

    def calculate_weight(self, hospital_address: str) -> int:
        return self.contract.functions.calculateWeight(
            Web3.to_checksum_address(hospital_address)
        ).call()
