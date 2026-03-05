"""
Hospital Node — Federated Learning Client
==========================================
Entry point for a hospital participant in the decentralized federated
learning network.  Each hospital node:

1. Registers itself on the smart contract.
2. Waits for a new training round to start.
3. Trains a local model on private data.
4. Submits the model-weight hash to the blockchain.
"""
