# FlyNode GPU Miner

A high-performance NVIDIA GPU CUDA Miner for [FlyNode.fun](https://flynode.fun).

It leverages `cupy` to compile a raw CUDA C++ Keccak-256 kernel on the fly, allowing extremely fast parallel nonce generation.

## Requirements
- NVIDIA GPU with CUDA Toolkit installed
- Python 3.8+

## Setup

```bash
pip install -r requirements.txt
```
*(Make sure to install the correct version of `cupy-cuda12x` or `cupy-cuda11x` depending on your CUDA installation.)*

## Usage

Intercept the mining parameters from your browser and run the script:

```bash
python gpu_miner.py --miner 0xYourAddress \
                    --prev 0xPreviousBlockHash \
                    --anchor 0xAnchorBlockHash \
                    --typeid 5 \
                    --target 15
```

The miner will output the winning nonce which you can then submit to the smart contract.

## Notes
- `target` is the number of leading zero bits required by the contract.
- The GPU will hash millions of nonces per second and return the first valid one.
