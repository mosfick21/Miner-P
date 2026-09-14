# HashGoat GPU Auto-Miner

Continuous NVIDIA CUDA miner for [HashGoat](https://www.hashgoat.fun/mine) on Robinhood Chain.

## Run

```bash
python -m pip install -r requirements.txt
python gpu_miner.py
```

The script asks for `PRIVATE_KEY` and an optional premium HTTP RPC URL using hidden terminal input. Neither is saved to a file, environment variable, or log. It discovers the public site configuration, verifies the chain and contract, uses the NVIDIA GPU, CPU-verifies every winning nonce, rejects stale challenges, and continues until `Ctrl+C` or the paid-mint limit is reached.

At startup, set `MAX MINT PRICE ETH` to `0` for free-only mode or enter the most you authorize per mint. Paid mode defaults to one successful mint; enter `0` for unlimited only if that is intentional. The price guard is checked again before every job.

Premium RPC routes are placed first, health-checked, and connection-warmed. Give the full provider HTTPS endpoint (including its API key if the provider embeds it in the URL). Public fallback routes remain enabled for parallel raw-transaction broadcast.

Verified protocol:

```text
chain:      Robinhood Chain (4663)
contract:   0x92102325e0B5Ef57709b783b8FF0C55e8f715736
hash:       SHA256(address(20) || uint256 nonce(32) || challenge(32))
valid:      leadingZeroBits(hash) >= currentDifficulty()
submission: mine(uint256 nonce, bytes32 challenge), selector 0xe43e322c
difficulty: refreshed automatically for every new challenge
```

The bot has FREE-only protection: it stops instead of submitting if `mintPrice()` becomes nonzero. Mining itself is probabilistic, so valid code cannot guarantee beating every competing miner.
