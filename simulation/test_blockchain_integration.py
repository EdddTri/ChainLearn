"""
test_blockchain_integration.py -- FLCoordinator integration smoke test
=====================================================================
Prerequisites:
1. Start Hardhat node
2. Deploy FLCoordinator
3. Set CONTRACT_ADDRESS
"""

import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def test_registration_and_submission():
    from hospital_node.blockchain_client import BlockchainClient
    from hospital_node.capacity_manager import compute_benchmark_hash, sign_benchmark_hash

    contract_address = os.environ.get("CONTRACT_ADDRESS", "")
    if not contract_address:
        print("CONTRACT_ADDRESS not set. Skipping integration test.")
        return

    abi_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "smart_contracts",
        "artifacts",
        "contracts",
        "FLCoordinator.sol",
        "FLCoordinator.json",
    )

    if not os.path.exists(abi_path):
        print(f"ABI not found at {abi_path}. Run `npx hardhat compile` first.")
        return

    owner_key = _require_env("FL_OWNER_PRIVATE_KEY")
    hospital_key = _require_env("FL_HOSPITAL1_PRIVATE_KEY")

    owner = BlockchainClient(
        provider_url="http://127.0.0.1:8545",
        contract_address=contract_address,
        abi_path=abi_path,
        private_key=owner_key,
    )

    hospital = BlockchainClient(
        provider_url="http://127.0.0.1:8545",
        contract_address=contract_address,
        abi_path=abi_path,
        private_key=hospital_key,
    )

    benchmark_hash = compute_benchmark_hash(123.4, 50, 16, "Medium")
    benchmark_sig = sign_benchmark_hash(benchmark_hash, hospital_key)

    print("1. Registering hospital...")
    owner.register_hospital(
        hospital_address=hospital.address,
        name="Integration Hospital",
        capacity_class=1,
        benchmark_hash=benchmark_hash,
        benchmark_signature=benchmark_sig,
    )

    print("2. Starting round...")
    owner.start_new_round()

    print("3. Submitting update...")
    model_hash = bytes.fromhex("11" * 32)
    hospital.submit_update(model_hash, confidence=9000, ece=500, model_type=1)  # Medium

    count = owner.get_hospital_count()
    weight = owner.calculate_weight(hospital.address)
    print(f"Hospital count: {count}")
    print(f"Weight: {weight}")

    print("Integration test passed.")


if __name__ == "__main__":
    test_registration_and_submission()
