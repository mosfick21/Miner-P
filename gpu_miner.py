import cupy as cp
import numpy as np
from web3 import Web3
import eth_abi.packed
import time
import argparse

CUDA_CODE = """
#include <cstdint>

class Keccak256 {
public:
    static constexpr int HASH_LEN = 32;
    __device__ static void getHash(const std::uint8_t msg[], std::size_t len, std::uint8_t hashResult[HASH_LEN]);
private:
    __device__ static void absorb(std::uint64_t state[5][5]);
    __device__ static std::uint64_t rotl64(std::uint64_t x, int i);
};

#define UINT64_C(c) (c ## ULL)

constexpr int Keccak256::HASH_LEN;
constexpr int NUM_ROUNDS = 24;

__constant__ unsigned char ROTATION[5][5] = {
    { 0, 36,  3, 41, 18},
    { 1, 44, 10, 45,  2},
    {62,  6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39,  8, 14}
};

__device__ __forceinline__ std::uint64_t Keccak256::rotl64(std::uint64_t x, int i) {
    return ((0U + x) << i) | (x >> ((64 - i) & 63));
}

__device__ __forceinline__ void Keccak256::absorb(uint64_t state[5][5]) {
    uint64_t (*a)[5] = state;
    uint8_t r = 1;  // LFSR
    for (int i = 0; i < NUM_ROUNDS; i++) {
        uint64_t c[5] = {};
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) c[x] ^= a[x][y];
        }
        for (int x = 0; x < 5; x++) {
            uint64_t d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
            for (int y = 0; y < 5; y++) a[x][y] ^= d;
        }
        uint64_t b[5][5];
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) b[y][(x * 2 + y * 3) % 5] = rotl64(a[x][y], ROTATION[x][y]);
        }
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) a[x][y] = b[x][y] ^ (~b[(x + 1) % 5][y] & b[(x + 2) % 5][y]);
        }
        for (int j = 0; j < 7; j++) {
            a[0][0] ^= static_cast<uint64_t>(r & 1) << ((1 << j) - 1);
            r = static_cast<uint8_t>((r << 1) ^ ((r >> 7) * 0x171));
        }
    }
}

__device__ __forceinline__ void Keccak256::getHash(const uint8_t msg[], size_t len, uint8_t hashResult[Keccak256::HASH_LEN]) {
    uint64_t state[5][5] = {};
    int blockOff = 0;
    const int BLOCK_SIZE = 200 - Keccak256::HASH_LEN * 2;
    for (size_t i = 0; i < len; i++) {
        int j = blockOff >> 3;
        state[j % 5][j / 5] ^= static_cast<uint64_t>(msg[i]) << ((blockOff & 7) << 3);
        blockOff++;
        if (blockOff == BLOCK_SIZE) {
            absorb(state);
            blockOff = 0;
        }
    }
    {
        int i = blockOff >> 3;
        state[i % 5][i / 5] ^= UINT64_C(0x01) << ((blockOff & 7) << 3);
        blockOff = BLOCK_SIZE - 1;
        int j = blockOff >> 3;
        state[j % 5][j / 5] ^= UINT64_C(0x80) << ((blockOff & 7) << 3);
        absorb(state);
    }
    for (int i = 0; i < Keccak256::HASH_LEN; i++) {
        int j = i >> 3;
        hashResult[i] = static_cast<uint8_t>(state[j % 5][j / 5] >> ((i & 7) << 3));
    }
}

extern "C" __global__ void mine_flynode(const uint8_t* base_payload, uint64_t start_nonce, int target_zeros, uint64_t* result_nonce) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t my_nonce = start_nonce + idx;
    
    // Copy the 118-byte base payload
    uint8_t msg[118];
    for(int i = 0; i < 118; i++) msg[i] = base_payload[i];
    
    // Inject the nonce into msg. The nonce is a uint256 (32 bytes) at offset 20.
    // In ABI encoding, uint256 is big-endian.
    // We vary the lower 8 bytes (uint64_t) of the nonce.
    // So the last 8 bytes of the nonce are at offset 20 + 24 = 44 to 51.
    for(int i = 0; i < 8; i++) {
        msg[51 - i] = (my_nonce >> (i * 8)) & 0xFF;
    }
    
    uint8_t hash[32];
    Keccak256::getHash(msg, 118, hash);
    
    // Check target zeros
    int zeros = 0;
    for(int i = 0; i < 32; i++) {
        uint8_t b = hash[i];
        if (b == 0) {
            zeros += 8;
        } else {
            // Count leading zeros of the byte
            for(int bit = 7; bit >= 0; bit--) {
                if ((b & (1 << bit)) == 0) zeros++;
                else break;
            }
            break;
        }
    }
    
    if (zeros >= target_zeros) {
        atomicExch((unsigned long long*)result_nonce, (unsigned long long)my_nonce);
    }
}
"""

def compile_gpu_miner():
    print("[*] Compiling CUDA kernel...")
    module = cp.RawModule(code=CUDA_CODE)
    return module.get_function('mine_flynode')

def mine_gpu(miner_address, prev, anchor, type_id, target_zeros, max_iters=1000):
    mine_kernel = compile_gpu_miner()
    
    # Initialize base payload with nonce=0
    print(f"[*] Preparing base payload for miner {miner_address}")
    base_payload = eth_abi.packed.encode_packed(
        ['address', 'uint256', 'bytes32', 'bytes32', 'uint16'],
        [miner_address, 0, prev, anchor, type_id]
    )
    
    d_base_payload = cp.array(list(base_payload), dtype=cp.uint8)
    d_result_nonce = cp.zeros(1, dtype=cp.uint64)
    
    threads_per_block = 256
    blocks_per_grid = 1024
    hashes_per_iter = threads_per_block * blocks_per_grid
    
    print(f"[*] Launching GPU miner. Target: {target_zeros} zeros.")
    
    start_time = time.time()
    for i in range(max_iters):
        start_nonce = i * hashes_per_iter
        
        # Launch kernel
        mine_kernel((blocks_per_grid,), (threads_per_block,), (d_base_payload, cp.uint64(start_nonce), cp.int32(target_zeros), d_result_nonce))
        
        # Check result
        res = int(d_result_nonce[0])
        if res != 0:
            end_time = time.time()
            elapsed = end_time - start_time
            total_hashes = start_nonce + hashes_per_iter
            hashrate = total_hashes / elapsed
            print(f"[+] FOUND NONCE: {res}")
            print(f"[+] Elapsed: {elapsed:.2f}s | Hashrate: {hashrate/1e6:.2f} MH/s")
            return res
    
    print("[-] No nonce found in the given iterations.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyNode GPU Miner")
    parser.add_argument("--miner", type=str, required=True, help="Miner address")
    parser.add_argument("--prev", type=str, required=True, help="Previous block hash (hex)")
    parser.add_argument("--anchor", type=str, required=True, help="Anchor block hash (hex)")
    parser.add_argument("--typeid", type=int, required=True, help="Type ID (uint16)")
    parser.add_argument("--target", type=int, default=15, help="Target zeros")
    
    args = parser.parse_args()
    
    prev_bytes = bytes.fromhex(args.prev.replace("0x", ""))
    anchor_bytes = bytes.fromhex(args.anchor.replace("0x", ""))
    
    mine_gpu(args.miner, prev_bytes, anchor_bytes, args.typeid, args.target)
