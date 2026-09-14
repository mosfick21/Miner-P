# HashGoat GPU Auto-Miner

Continuous NVIDIA CUDA miner for [HashGoat](https://www.hashgoat.fun/mine) on Robinhood Chain.

## Run

```bash
python -m pip install -r requirements.txt
python gpu_miner.py
```

The script asks only for `PRIVATE_KEY` using hidden terminal input. It never saves the key to a file, config, environment variable, or log. It discovers the public site configuration, verifies the chain and contract, uses the full NVIDIA GPU, CPU-verifies every winning nonce, rejects stale challenges, submits the free mint, and continues until `Ctrl+C`.

Verified protocol:

```text
chain:      Robinhood Chain (4663)
contract:   0x92102325e0B5Ef57709b783b8FF0C55e8f715736
hash:       SHA256(address(20) || uint256 nonce(32) || challenge(32))
valid:      leadingZeroBits(hash) >= currentDifficulty()
submission: mine(uint256 nonce, bytes32 challenge), selector 0xe43e322c
gate:       lastMintAt() + 15 seconds, selector 0xda444f0b
```

The bot has FREE-only protection: it stops instead of submitting if `mintPrice()` becomes nonzero. Mining itself is probabilistic, so valid code cannot guarantee beating every competing miner.
