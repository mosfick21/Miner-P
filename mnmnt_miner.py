#!/usr/bin/env python3
"""MNMNT CUDA free miner. The prompted private key is never saved."""

from __future__ import annotations

import getpass
import os
import re
import secrets
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np
from eth_account import Account
from eth_utils import keccak, to_checksum_address
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ranger_miner import CUDA_SOURCE as KECCAK_CUDA_SOURCE
from ranger_miner import GPU as KeccakGPU
from ranger_miner import RpcPool, n, word


CHAIN_ID = 4663
CHAIN_NAME = "Robinhood Chain"
CONTRACT = to_checksum_address("0xd2E3762Ae59B77AE42885E17fCCA10F219Ac4872")
RPC_URLS = [
    "https://rpc.mainnet.chain.robinhood.com",
    "https://robinhood-rpc.publicnode.com",
    "https://rpc.ordofi.network",
]
SUPPLY = 8190
STATE = "0xc19d93fb"                 # state()
LAY = "0x517ec447"                   # lay(uint256,uint256)
POLL_SECONDS = 0.25
# Aggressive GPU mode: keep each launch busy long enough to remove notebook /
# Python launch overhead. There is deliberately no sleep or utilization cap in
# the mining loop.
TARGET_BATCH_SECONDS = 0.75
GAS_LIMIT = 800_000
MIN_PRIORITY_FEE = 20_000_000


# ranger_miner has an optimized one-block Keccak-256 kernel. MNMNT's packed
# payload is seed[32] || sender[20] || uint256 nonce[32], so the variable
# low 64 nonce bits live in lanes 9/10 instead of lanes 5/6.
CUDA_SOURCE = KECCAK_CUDA_SOURCE.replace(
    " s[5]=((uint64_t)sw32(hi))<<32;\n s[6]=(s[6]&0xffffffff00000000ULL)|(uint64_t)sw32(lo);",
    " s[9]=(s[9]&0x00000000ffffffffULL)|(((uint64_t)sw32(hi))<<32);\n"
    " s[10]=(s[10]&0xffffffff00000000ULL)|(uint64_t)sw32(lo);",
)
if CUDA_SOURCE == KECCAK_CUDA_SOURCE:
    raise RuntimeError("Could not configure MNMNT CUDA payload")


@dataclass
class Job:
    laid: int
    seed: str
    open_at: int
    price: int
    target: int
    course: int
    tx_nonce: int = 0
    gas_price: int = 0
    base_fee: int | None = None
    balance: int = 0


def decode_state(raw: str) -> Job:
    if not raw or raw == "0x":
        raise RuntimeError("state() returned empty data")
    data = bytes.fromhex(raw[2:])
    if len(data) < 16 * 32:
        raise RuntimeError(f"state() returned only {len(data)} bytes")
    values = [int.from_bytes(data[i : i + 32], "big") for i in range(0, 16 * 32, 32)]
    return Job(
        laid=values[0],
        seed="0x" + data[32:64].hex(),
        open_at=values[2],
        price=values[3],
        target=values[4],
        course=values[5],
    )


def state_call() -> tuple[str, list[Any]]:
    return "eth_call", [{"to": CONTRACT, "data": STATE}, "latest"]


def read_state(rpc: RpcPool) -> Job:
    return decode_state(rpc.call(*state_call()))


def get_job(rpc: RpcPool, address: str) -> Job:
    raw, tx_nonce, gas_price, block, balance = rpc.batch(
        [
            state_call(),
            ("eth_getTransactionCount", [address, "pending"]),
            ("eth_gasPrice", []),
            ("eth_getBlockByNumber", ["latest", False]),
            ("eth_getBalance", [address, "latest"]),
        ]
    )
    job = decode_state(raw)
    job.tx_nonce = n(tx_nonce)
    job.gas_price = n(gas_price)
    job.base_fee = int(block["baseFeePerGas"], 16) if block.get("baseFeePerGas") else None
    job.balance = n(balance)
    return job


def payload(address: str, nonce: int, seed: str) -> bytes:
    return bytes.fromhex(seed[2:]) + bytes.fromhex(address[2:]) + nonce.to_bytes(32, "big")


def digest(address: str, nonce: int, seed: str) -> bytes:
    return keccak(payload(address, nonce, seed))


def leading_zero_bits(value: bytes) -> int:
    count = 0
    for byte in value:
        if byte == 0:
            count += 8
        else:
            count += 8 - byte.bit_length()
            break
    return count


def base_lanes(address: str, seed: str) -> np.ndarray:
    raw = bytearray(136)
    raw[0:32] = bytes.fromhex(seed[2:])
    raw[32:52] = bytes.fromhex(address[2:])
    # bytes 52..83 are the big-endian uint256 nonce (upper 192 bits stay zero)
    raw[84] ^= 0x01
    raw[135] ^= 0x80
    return np.asarray(
        [int.from_bytes(raw[i : i + 8], "little") for i in range(0, 136, 8)],
        dtype=np.uint64,
    )


def target_words(target: int) -> np.ndarray:
    raw = target.to_bytes(32, "big")
    return np.asarray([int.from_bytes(raw[i : i + 4], "big") for i in range(0, 32, 4)], dtype=np.uint32)


class GPU:
    tune = KeccakGPU.tune

    def __init__(self, device_id: int) -> None:
        self.device_id = device_id
        with cp.cuda.Device(device_id):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            name = props.get("name", b"NVIDIA GPU")
            self.name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
            self.sms = max(1, int(props.get("multiProcessorCount", 1)))
            self.blocks = max(64, self.sms * 8)
            self.threads = 256
            self.iters = 128
            self.tuned_rate = 0.0
            module = cp.RawModule(
                code=CUDA_SOURCE,
                options=("--std=c++14", "--use_fast_math"),
                name_expressions=("ranger_mine", "ranger_hash_one"),
            )
            self.kernel = module.get_function("ranger_mine")
            self.one = module.get_function("ranger_hash_one")
            self.found = cp.zeros(1, cp.uint32)
            self.answer = cp.zeros(1, cp.uint64)
            self.best = cp.zeros(1, cp.uint32)
            self.bestnonce = cp.zeros(1, cp.uint64)
            self.base = None
            self.target = None

    def test(self, address: str, job: Job) -> None:
        with cp.cuda.Device(self.device_id):
            base = cp.asarray(base_lanes(address, job.seed))
            out = cp.zeros(8, cp.uint32)
            for nonce in (0, 1, 0x1122334455667788):
                self.one(
                    (1,),
                    (1,),
                    (base, np.uint32(nonce & 0xFFFFFFFF), np.uint32(nonce >> 32), out),
                )
                got = b"".join(int(x).to_bytes(4, "big") for x in cp.asnumpy(out))
                if got != digest(address, nonce, job.seed):
                    raise RuntimeError(f"GPU #{self.device_id + 1} Keccak self-test failed")
            self.tune(base)
            lanes = self.blocks * self.threads
            self.iters = max(16, min(65535, int(TARGET_BATCH_SECONDS * self.tuned_rate / lanes)))

    def prepare(self, address: str, job: Job) -> None:
        with cp.cuda.Device(self.device_id):
            self.base = cp.asarray(base_lanes(address, job.seed))
            self.target = cp.asarray(target_words(job.target))

    def batch(self, first: int) -> dict[str, Any]:
        with cp.cuda.Device(self.device_id):
            self.found.fill(0); self.answer.fill(0); self.best.fill(0); self.bestnonce.fill(0)
            hashes = self.blocks * self.threads * self.iters
            began = time.perf_counter()
            self.kernel(
                (self.blocks,), (self.threads,),
                (self.base, self.target, np.uint32(first & 0xFFFFFFFF), np.uint32(first >> 32),
                 np.uint32(self.iters), self.found, self.answer, self.best, self.bestnonce),
            )
            cp.cuda.Stream.null.synchronize()
            elapsed = max(time.perf_counter() - began, 0.001)
            result = {
                "device": self.device_id,
                "hit": int(cp.asnumpy(self.found)[0]),
                "nonce": int(cp.asnumpy(self.answer)[0]),
                "best": int(cp.asnumpy(self.best)[0]),
                "bestnonce": int(cp.asnumpy(self.bestnonce)[0]),
                "hashes": hashes,
                "elapsed": elapsed,
                "rate": hashes / elapsed,
            }
            self.iters = max(16, min(65535, int(self.iters * min(1.35, max(0.75, TARGET_BATCH_SECONDS / elapsed)))))
            return result

    def mine(self, address: str, job: Job, rpc: RpcPool, ui: "UI", session: int) -> tuple[str, int | None, int]:
        base = cp.asarray(base_lanes(address, job.seed))
        target = cp.asarray(target_words(job.target))
        start = secrets.randbits(64)
        offset = count = best = 0
        rate = 0.0
        best_hash = "-"
        probe = None
        next_poll = 0.0
        watcher = ThreadPoolExecutor(max_workers=1)
        watch_rpc = RpcPool(rpc.urls)
        try:
            while True:
                now = time.monotonic()
                if probe is None and now >= next_poll:
                    probe = watcher.submit(read_state, watch_rpc)
                    next_poll = now + POLL_SECONDS

                self.found.fill(0)
                self.answer.fill(0)
                self.best.fill(0)
                self.bestnonce.fill(0)
                first = (start + offset) & ((1 << 64) - 1)
                hashes = self.blocks * self.threads * self.iters
                began = time.perf_counter()
                self.kernel(
                    (self.blocks,),
                    (self.threads,),
                    (
                        base,
                        target,
                        np.uint32(first & 0xFFFFFFFF),
                        np.uint32(first >> 32),
                        np.uint32(self.iters),
                        self.found,
                        self.answer,
                        self.best,
                        self.bestnonce,
                    ),
                )
                cp.cuda.Stream.null.synchronize()
                elapsed = max(time.perf_counter() - began, 0.001)
                hit = int(cp.asnumpy(self.found)[0])
                nonce = int(cp.asnumpy(self.answer)[0])
                batch_best = int(cp.asnumpy(self.best)[0])
                best_nonce = int(cp.asnumpy(self.bestnonce)[0])
                instant = hashes / elapsed
                rate = instant if not rate else rate * 0.72 + instant * 0.28
                count += hashes
                offset = (offset + hashes) & ((1 << 64) - 1)
                if batch_best > best:
                    best = batch_best
                    best_hash = "0x" + digest(address, best_nonce, job.seed).hex()
                ui.data.update(
                    phase="MINING",
                    rate=rate,
                    job_hashes=count,
                    session_hashes=session + count,
                    best=best,
                    besthash=best_hash,
                    batch_ms=elapsed * 1000,
                )
                ui.refresh()
                self.iters = max(16, min(65535, int(self.iters * min(1.45, max(0.70, TARGET_BATCH_SECONDS / elapsed)))))

                if probe is not None and probe.done():
                    try:
                        current = probe.result()
                        if current.seed != job.seed or current.laid >= SUPPLY:
                            return "stale", None, count
                        if current.target != job.target:
                            return "stale", None, count
                    except Exception as exc:
                        ui.log(f"RPC monitor retry: {exc}", "yellow")
                    probe = None

                if hit:
                    proof = digest(address, nonce, job.seed)
                    if int.from_bytes(proof, "big") >= job.target:
                        raise RuntimeError("GPU nonce failed CPU verification")
                    return "found", nonce, count
        finally:
            watcher.shutdown(wait=False, cancel_futures=True)


class GPUFarm:
    """Run one independent nonce lane on every visible CUDA GPU."""

    def __init__(self) -> None:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            raise RuntimeError("No CUDA GPU found")
        self.gpus = [GPU(i) for i in range(count)]
        self.name = " + ".join(f"#{i + 1} {gpu.name}" for i, gpu in enumerate(self.gpus))
        self.tuned_rate = 0.0

    def test(self, address: str, job: Job) -> None:
        for gpu in self.gpus:
            gpu.test(address, job)
        self.tuned_rate = sum(gpu.tuned_rate for gpu in self.gpus)

    def mine(self, address: str, job: Job, rpc: RpcPool, ui: "UI", session: int) -> tuple[str, int | None, int]:
        for gpu in self.gpus:
            gpu.prepare(address, job)
        starts = [secrets.randbits(64) for _ in self.gpus]
        offsets = [0 for _ in self.gpus]
        count = best = 0
        best_hash = "-"
        pool = ThreadPoolExecutor(max_workers=len(self.gpus))
        monitor_pool = ThreadPoolExecutor(max_workers=1)
        watch_rpc = RpcPool(rpc.urls)
        try:
            while True:
                state_future = monitor_pool.submit(read_state, watch_rpc)
                futures = [
                    pool.submit(gpu.batch, (starts[i] + offsets[i]) & ((1 << 64) - 1))
                    for i, gpu in enumerate(self.gpus)
                ]
                results = [future.result() for future in futures]
                batch_hashes = sum(result["hashes"] for result in results)
                count += batch_hashes
                for i, result in enumerate(results):
                    offsets[i] = (offsets[i] + result["hashes"]) & ((1 << 64) - 1)
                    if result["best"] > best:
                        best = result["best"]
                        best_hash = "0x" + digest(address, result["bestnonce"], job.seed).hex()
                total_rate = sum(result["rate"] for result in results)
                ui.data.update(
                    phase="MINING", rate=total_rate, job_hashes=count,
                    session_hashes=session + count, best=best, besthash=best_hash,
                    batch_ms=max(result["elapsed"] for result in results) * 1000,
                )
                ui.refresh()

                current = None
                try:
                    current = state_future.result(timeout=2)
                except TimeoutError:
                    pass
                except Exception as exc:
                    ui.log(f"RPC monitor retry: {exc}", "yellow")

                for result in results:
                    if result["hit"]:
                        if current is not None and (current.seed != job.seed or current.laid >= SUPPLY):
                            continue
                        nonce = result["nonce"]
                        proof = digest(address, nonce, job.seed)
                        live_target = current.target if current is not None and current.seed == job.seed else job.target
                        if int.from_bytes(proof, "big") >= live_target:
                            continue
                        if int.from_bytes(proof, "big") >= job.target:
                            raise RuntimeError(f"GPU #{result['device'] + 1} nonce failed CPU verification")
                        return "found", nonce, count
                if current is not None and (
                    current.seed != job.seed or current.laid >= SUPPLY or current.target != job.target
                ):
                    return "stale", None, count
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            monitor_pool.shutdown(wait=False, cancel_futures=True)


class UI:
    def __init__(self, address: str, gpu: str, routes: int) -> None:
        self.address = address
        self.gpu = gpu
        self.routes = routes
        self.logs: deque[tuple[str, str]] = deque(maxlen=6)
        self.live: Live | None = None
        self.data = {
            "phase": "STARTING", "laid": 0, "price": 0, "target": 0,
            "rate": 0.0, "job_hashes": 0, "session_hashes": 0,
            "best": 0, "besthash": "-", "batch_ms": 0.0,
            "mints": 0, "tx": "-", "course": 0,
        }

    @staticmethod
    def short(value: str) -> str:
        return value if len(value) < 35 else value[:18] + "..." + value[-12:]

    @staticmethod
    def rate(value: float) -> str:
        return f"{value / 1e9:.2f} GH/s" if value >= 1e9 else f"{value / 1e6:.2f} MH/s"

    @staticmethod
    def count(value: int) -> str:
        return f"{value / 1e9:.2f}B" if value >= 10**9 else f"{value / 1e6:.2f}M" if value >= 10**6 else f"{value:,}"

    def log(self, message: str, color: str = "cyan") -> None:
        self.logs.append((time.strftime("%H:%M:%S  ") + message, color))
        self.refresh()

    def render(self) -> Group:
        d = self.data
        top = Table.grid(expand=True)
        top.add_column(style="bold cyan", width=12); top.add_column()
        top.add_column(style="bold cyan", width=12); top.add_column()
        top.add_row("NETWORK", f"{CHAIN_NAME} ({CHAIN_ID})", "PHASE", d["phase"])
        top.add_row("WALLET", self.short(self.address), "GPU", self.gpu)
        top.add_row("CONTRACT", self.short(CONTRACT), "MODE", "FREE ONLY (0 ETH)")
        top.add_row("LAID", f"{d['laid']:,} / {SUPPLY:,}", "COURSE", str(d["course"]))
        top.add_row("RPC ROUTES", str(self.routes), "MARKET PRICE", f"{d['price'] / 1e18:.6f} ETH (not paid)")

        mine = Table.grid(expand=True)
        mine.add_column(style="bright_green", width=14); mine.add_column()
        mine.add_row("HASHRATE", self.rate(d["rate"]))
        mine.add_row("BEST", f"{d['best']} leading-zero bits")
        mine.add_row("HASHES", f"job {self.count(d['job_hashes'])} | session {self.count(d['session_hashes'])}")
        mine.add_row("GPU BATCH", f"{d['batch_ms']:.0f} ms")
        mine.add_row("TARGET", f"0x{d['target']:064x}" if d["target"] else "-")
        mine.add_row("BEST HASH", self.short(d["besthash"]))
        mine.add_row("MINTED", str(d["mints"]))
        mine.add_row("LAST TX", self.short(d["tx"]))

        lines = [Text(text, style=color) for text, color in self.logs] or [Text("Starting...", style="dim")]
        return Group(
            Panel(top, title="MNMNT GPU FREE MINER", border_style="bright_cyan"),
            Panel(mine, title="LIVE MINING", border_style="bright_green"),
            Panel(Group(*lines), title="ACTIVITY", border_style="blue"),
            Text(" Ctrl+C: stop safely | private key is memory-only ", style="bold black on bright_cyan"),
        )

    def refresh(self) -> None:
        if self.live:
            self.live.update(self.render(), refresh=True)


def lay_calldata(nonce: int) -> str:
    return LAY + word(0) + word(nonce)


def submit(rpc: RpcPool, account: Any, nonce: int, job: Job) -> str:
    # Refresh only transaction metadata. The caller already checked the current
    # target; value remains hardcoded to zero under every condition.
    tx_nonce, gas_price, block, balance = rpc.batch(
        [
            ("eth_getTransactionCount", [account.address, "pending"]),
            ("eth_gasPrice", []),
            ("eth_getBlockByNumber", ["latest", False]),
            ("eth_getBalance", [account.address, "latest"]),
        ]
    )
    gas_price_i = n(gas_price)
    base_fee = int(block["baseFeePerGas"], 16) if block.get("baseFeePerGas") else None
    if base_fee is None:
        unit = max(gas_price_i * 125 // 100, MIN_PRIORITY_FEE)
        fees = {"gasPrice": unit}
    else:
        network_tip = max(0, gas_price_i - base_fee)
        tip = max(network_tip * 125 // 100, MIN_PRIORITY_FEE)
        unit = base_fee * 2 + tip
        fees = {"type": 2, "maxFeePerGas": unit, "maxPriorityFeePerGas": tip}
    if n(balance) < GAS_LIMIT * unit:
        raise RuntimeError(f"Insufficient gas ETH; need up to {GAS_LIMIT * unit / 1e18:.8f} ETH")
    tx = {
        "chainId": CHAIN_ID,
        "nonce": n(tx_nonce),
        "to": CONTRACT,
        "value": 0,
        "data": lay_calldata(nonce),
        "gas": GAS_LIMIT,
        **fees,
    }
    signed = account.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    return rpc.broadcast("0x" + bytes(raw).hex(), "0x" + bytes(signed.hash).hex())


def wait_receipt(rpc: RpcPool, tx_hash: str) -> dict[str, Any]:
    until = time.monotonic() + 180
    while time.monotonic() < until:
        result = rpc.call("eth_getTransactionReceipt", [tx_hash])
        if result:
            return result
        time.sleep(0.35)
    raise TimeoutError("Receipt timeout")


def load_account() -> Any:
    key = getpass.getpass("PRIVATE_KEY (hidden, never saved): ").strip()
    if not key:
        raise ValueError("Empty private key")
    if not key.startswith("0x"):
        key = "0x" + key
    try:
        return Account.from_key(key)
    finally:
        key = ""


def rpc_urls() -> list[str]:
    premium = os.environ.get("MNMNT_RPC_URLS", "").strip()
    extra = [url.rstrip("/") for url in re.split(r"[,\s]+", premium) if url]
    if any(not re.match(r"^https?://", url, re.I) for url in extra):
        raise ValueError("MNMNT_RPC_URLS contains a non-HTTP URL")
    return list(dict.fromkeys(extra + RPC_URLS))


def run() -> int:
    console = Console()
    console.print("[bold cyan]MNMNT GPU Free Miner[/bold cyan]")
    try:
        account = load_account()
        rpc = RpcPool(rpc_urls())
        warm = rpc.warmup()
        job = get_job(rpc, account.address)
        if job.seed == "0x" + "00" * 32:
            raise RuntimeError(f"Contract is not open yet (openAt {job.open_at})")
        gpu = GPUFarm()
        gpu.test(account.address, job)
    except Exception as exc:
        console.print(f"[red]Startup failed: {exc}[/red]")
        return 1

    ui = UI(account.address, gpu.name, len(rpc.urls))
    session_hashes = 0
    with Live(ui.render(), console=console, refresh_per_second=4, screen=False) as live:
        ui.live = live
        ui.log(f"{len(gpu.gpus)} GPU(s) detected; all Keccak self-tests passed", "green")
        ui.log(f"ALL-GPU mode | combined benchmark {UI.rate(gpu.tuned_rate)}", "green")
        ui.log("RPC warm: " + " | ".join(f"#{i + 1} {ms:.0f}ms" for i, ms in enumerate(warm)), "green")
        ui.log("FREE-only lock active: every lay transaction sends 0 ETH", "bright_green")
        try:
            while True:
                job = get_job(rpc, account.address)
                ui.data.update(
                    phase="READY", laid=job.laid, price=job.price, target=job.target,
                    course=job.course, job_hashes=0, best=0, besthash="-",
                )
                ui.refresh()
                if job.laid >= SUPPLY:
                    ui.log("Collection complete", "yellow")
                    break

                ui.log(f"Mining stone #{job.laid + 2} at current target", "bright_green")
                status, nonce, used_hashes = gpu.mine(account.address, job, rpc, ui, session_hashes)
                session_hashes += used_hashes
                ui.data["session_hashes"] = session_hashes
                if status == "stale":
                    ui.log("Target changed; switched to fresh state with 0 gas", "yellow")
                    continue

                # Mandatory final state check. A proof remains usable only if it
                # is below the latest, possibly tighter target.
                current = get_job(rpc, account.address)
                proof = digest(account.address, int(nonce), current.seed)
                if current.seed != job.seed or int.from_bytes(proof, "big") >= current.target:
                    ui.log("Proof became stale before submission; mining fresh target", "yellow")
                    continue

                ui.data.update(phase="SUBMITTING", laid=current.laid, target=current.target, price=current.price, course=current.course)
                ui.log(f"Valid free proof; immediate {len(rpc.urls)}-route broadcast", "green")
                ui.refresh()
                tx_hash = submit(rpc, account, int(nonce), current)
                ui.data.update(phase="CONFIRMING", tx=tx_hash)
                ui.log(f"Submitted {ui.short(tx_hash)}")
                receipt = wait_receipt(rpc, tx_hash)
                gas = n(receipt.get("gasUsed", "0x0"))
                if n(receipt.get("status", "0x0")) == 1:
                    ui.data["mints"] += 1
                    ui.log(f"FREE MINT SUCCESS | gas {gas:,}", "bold green")
                else:
                    ui.log(f"Reverted after state moved | gas {gas:,}; continuing", "red")
                ui.refresh()
        except KeyboardInterrupt:
            ui.log("Stopped safely by user", "yellow")
        except Exception as exc:
            ui.data["phase"] = "ERROR"
            ui.log(str(exc), "red")
            return 1
        finally:
            ui.live = None
    return 0


if __name__ == "__main__":
    sys.exit(run())
