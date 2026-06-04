"""JAX port of the training pipeline.

Modules:
- nets.py: Flax modules for the PRENet policy/value network and helpers
- trainer.py: A2C/GAE training loop with Optax
- utils.py: Env helpers and wrappers mirroring the PyTorch version
- train.py: CLI entrypoint to run training
"""
