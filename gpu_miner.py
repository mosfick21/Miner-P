#!/usr/bin/env python3
"""HashGoat CUDA auto-miner. The prompted private key is never saved."""

from __future__ import annotations

import getpass
import hashlib
import re
import secrets
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import cupy as cp
import numpy as np
import requests
from eth_account import Account
from eth_utils import to_checksum_address
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SITE_CONFIG_URL = "https://www.hashgoat.fun/config.js"
FALLBACK = {
    "chain_id": 4663,
    "chain_name": "Robinhood Chain",
    "rpc_urls": ["https://rpc.mainnet.chain.robinhood.com/", "https://robinhood.drpc.org"],
    "contract": "0x92102325e0B5Ef57709b783b8FF0C55e8f715736",
}

TOTAL_SUPPLY = "0x18160ddd"
MINT_PRICE = "0x6817c76c"
CHALLENGE = "0xd2ef7398"
DIFFICULTY = "0x5c062d6c"
MINE = "0xe43e322c"  # mine(uint256,bytes32)
POLL_SECONDS = 0.20
TARGET_BATCH_SECONDS = 0.18
THREADS = 256
MINE_GAS_LIMIT = 160_000
GAS_BUMP_NUM = 125
GAS_BUMP_DEN = 100
MIN_PRIORITY_FEE = 20_000_000
LAST_MINE_BLOCK_SLOT = "0x" + "0" * 62 + "14"

CUDA_SOURCE = r"""
// NVRTC environments such as Kaggle may not expose host C headers.
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
__device__ __constant__ uint32_t K[64]={
0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
__device__ __forceinline__ uint32_t rr(uint32_t x,uint32_t n){return(x>>n)|(x<<(32u-n));}
__device__ __forceinline__ uint32_t ch(uint32_t x,uint32_t y,uint32_t z){return(x&y)^((~x)&z);}
__device__ __forceinline__ uint32_t maj(uint32_t x,uint32_t y,uint32_t z){return(x&y)^(x&z)^(y&z);}
__device__ __forceinline__ uint32_t b0(uint32_t x){return rr(x,2)^rr(x,13)^rr(x,22);}
__device__ __forceinline__ uint32_t b1(uint32_t x){return rr(x,6)^rr(x,11)^rr(x,25);}
__device__ __forceinline__ uint32_t s0(uint32_t x){return rr(x,7)^rr(x,18)^(x>>3);}
__device__ __forceinline__ uint32_t s1(uint32_t x){return rr(x,17)^rr(x,19)^(x>>10);}
__device__ __forceinline__ void comp(uint32_t st[8],const uint32_t bl[16]){
 uint32_t w[64];
 #pragma unroll
 for(int i=0;i<16;i++)w[i]=bl[i];
 #pragma unroll
 for(int i=16;i<64;i++)w[i]=s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16];
 uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
 #pragma unroll
 for(int i=0;i<64;i++){uint32_t t1=h+b1(e)+ch(e,f,g)+K[i]+w[i],t2=b0(a)+maj(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
 st[0]+=a;st[1]+=b;st[2]+=c;st[3]+=d;st[4]+=e;st[5]+=f;st[6]+=g;st[7]+=h;
}
// SHA256(address(20) || uint256 nonce(32) || bytes32 challenge(32))
__device__ __forceinline__ void digest(const uint32_t*base,uint32_t lo,uint32_t hi,uint32_t out[8]){
 uint32_t x[16];x[0]=base[0];x[1]=base[1];x[2]=base[2];x[3]=base[3];x[4]=base[4];
 x[5]=0;x[6]=0;x[7]=0;x[8]=0;x[9]=0;x[10]=0;x[11]=hi;x[12]=lo;x[13]=base[5];x[14]=base[6];x[15]=base[7];
 out[0]=0x6a09e667u;out[1]=0xbb67ae85u;out[2]=0x3c6ef372u;out[3]=0xa54ff53au;
 out[4]=0x510e527fu;out[5]=0x9b05688cu;out[6]=0x1f83d9abu;out[7]=0x5be0cd19u;comp(out,x);
 uint32_t y[16];y[0]=base[8];y[1]=base[9];y[2]=base[10];y[3]=base[11];y[4]=base[12];y[5]=0x80000000u;
 #pragma unroll
 for(int i=6;i<15;i++)y[i]=0;y[15]=672u;comp(out,y);
}
__device__ __forceinline__ uint32_t zeros(const uint32_t h[8]){uint32_t n=0;for(int i=0;i<8;i++){if(h[i]==0)n+=32;else{n+=__clz(h[i]);break;}}return n;}
extern "C" __global__ void hashgoat_mine(const uint32_t*base,uint32_t slo,uint32_t shi,uint32_t iters,uint32_t target,uint32_t*found,unsigned long long*answer,uint32_t*best,unsigned long long*bestnonce){
 uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x,stride=(uint64_t)gridDim.x*blockDim.x,start=((uint64_t)shi<<32)|slo;
 uint32_t local=0;uint64_t localnonce=start+tid;
 for(uint32_t i=0;i<iters;i++){if(__ldg(found))break;uint64_t nonce=start+tid+(uint64_t)i*stride;uint32_t h[8];digest(base,(uint32_t)nonce,(uint32_t)(nonce>>32),h);uint32_t z=zeros(h);if(z>local){local=z;localnonce=nonce;}if(z>=target){if(atomicCAS(found,0u,1u)==0u)*answer=nonce;break;}}
 uint32_t old=atomicMax(best,local);if(local>old)*bestnonce=localnonce;
}
extern "C" __global__ void hashgoat_hash_one(const uint32_t*base,uint32_t lo,uint32_t hi,uint32_t*out){if(blockIdx.x||threadIdx.x)return;uint32_t h[8];digest(base,lo,hi,h);for(int i=0;i<8;i++)out[i]=h[i];}
"""


class RpcError(RuntimeError):
    pass


class RpcRejected(RpcError):
    """A valid JSON-RPC response rejected the call; retrying cannot fix it."""


class RpcPool:
    def __init__(self, urls: list[str]):
        self.urls = list(dict.fromkeys(urls)); self.index = 0; self.ident = 0
        self.sessions = {u: requests.Session() for u in self.urls}
        self.broadcast_sessions = {u: requests.Session() for u in self.urls}
        self.broadcast_pool = ThreadPoolExecutor(max_workers=max(1, len(self.urls)))

    @staticmethod
    def _request(session: requests.Session, url: str, method: str, params: list[Any] | None = None, timeout: float = 12) -> Any:
        response=session.post(url,json={"jsonrpc":"2.0","id":int(time.time_ns()%2_000_000_000),"method":method,"params":params or []},timeout=timeout)
        response.raise_for_status();body=response.json()
        if body.get("error"):raise RpcRejected(body["error"].get("message",str(body["error"])))
        return body["result"]

    def warmup(self, chain_id: int) -> list[tuple[str,float]]:
        def probe(url: str) -> tuple[str,float]:
            began=time.perf_counter();value=self._request(self.broadcast_sessions[url],url,"eth_chainId",timeout=8)
            if int(value,16)!=chain_id:raise RpcError(f"chain {int(value,16)}")
            return url,(time.perf_counter()-began)*1000
        healthy=[];failures=[]
        futures={self.broadcast_pool.submit(probe,url):url for url in self.urls}
        for future in as_completed(futures):
            try:healthy.append(future.result())
            except Exception as exc:failures.append(f"{futures[future]}: {exc}")
        good={url for url,_ in healthy}
        self.urls=[url for url in self.urls if url in good]
        if not self.urls:raise RpcError("all RPC routes failed: "+" | ".join(failures))
        return sorted(healthy,key=lambda item:self.urls.index(item[0]))

    def call(self, method: str, params: list[Any] | None = None) -> Any:
        error: Exception | None = None
        for attempt in range(max(3, len(self.urls) * 2)):
            url = self.urls[(self.index + attempt) % len(self.urls)]; self.ident += 1
            try:
                response = self.sessions[url].post(url, json={"jsonrpc":"2.0","id":self.ident,"method":method,"params":params or []}, timeout=12)
                response.raise_for_status(); body = response.json()
                if body.get("error"): raise RpcRejected(body["error"].get("message", str(body["error"])))
                self.index = self.urls.index(url); return body["result"]
            except RpcRejected:
                raise
            except Exception as exc:
                error = exc; time.sleep(min(.15 * (attempt + 1), .75))
        raise RpcError(f"RPC failed: {error}")

    def batch(self, calls: list[tuple[str, list[Any]]]) -> list[Any]:
        error: Exception | None = None
        for attempt in range(max(3, len(self.urls) * 2)):
            url = self.urls[(self.index + attempt) % len(self.urls)]; payload=[]; ids=[]
            for method, params in calls:
                self.ident += 1; ids.append(self.ident)
                payload.append({"jsonrpc":"2.0","id":self.ident,"method":method,"params":params})
            try:
                response=self.sessions[url].post(url,json=payload,timeout=12);response.raise_for_status();items=response.json();by_id={x["id"]:x for x in items};out=[]
                for ident in ids:
                    item=by_id[ident]
                    if item.get("error"):raise RpcRejected(item["error"].get("message",str(item["error"])))
                    out.append(item["result"])
                self.index=self.urls.index(url);return out
            except RpcRejected:
                raise
            except Exception as exc:
                error=exc;time.sleep(min(.15*(attempt+1),.75))
        raise RpcError(f"RPC batch failed: {error}")

    def broadcast(self, raw_tx: str, expected_hash: str) -> str:
        def send(url: str) -> tuple[str, str]:
            session=self.broadcast_sessions[url];payload={"jsonrpc":"2.0","id":int(time.time_ns()%2_000_000_000),"method":"eth_sendRawTransaction","params":[raw_tx]}
            response=session.post(url,json=payload,timeout=12);response.raise_for_status();body=response.json()
            if body.get("result"):return "ok",body["result"]
            return "error",body.get("error",{}).get("message",str(body.get("error")))
        errors=[]
        futures=[self.broadcast_pool.submit(send,url) for url in self.urls]
        for future in as_completed(futures):
            try:
                status,value=future.result()
                if status=="ok":return value
                errors.append(value)
            except Exception as exc:errors.append(str(exc))
        if any("already known" in x.lower() or "nonce too low" in x.lower() for x in errors):return expected_hash
        raise RpcError("broadcast failed: "+" | ".join(errors))


@dataclass(frozen=True)
class Config:
    chain_id: int; chain_name: str; rpc_urls: list[str]; contract: str; source: str


@dataclass
class Job:
    supply: int; price: int; challenge: str; difficulty: int; tx_nonce: int; gas_price: int; base_fee: int|None; balance: int; latest_block: int; last_mine_block: int


@dataclass(frozen=True)
class Limits:
    max_price: int
    max_paid_mints: int


def discover() -> Config:
    try:
        text=requests.get(SITE_CONFIG_URL,timeout=12).text
        def take(pattern: str) -> str:
            match=re.search(pattern,text,re.I)
            if not match:raise ValueError(pattern)
            return match.group(1)
        contract=take(r"CONTRACT_ADDRESS\s*:\s*['\"](0x[0-9a-f]{40})")
        chain_id=int(take(r"CHAIN_ID_DECIMAL\s*:\s*(\d+)"));name=take(r"CHAIN_NAME\s*:\s*['\"]([^'\"]+)");rpc=take(r"RPC_URL\s*:\s*['\"]([^'\"]+)")
        urls=[rpc]+(FALLBACK["rpc_urls"] if chain_id==4663 else [])
        return Config(chain_id,name,list(dict.fromkeys(urls)),to_checksum_address(contract),"live site")
    except Exception:
        return Config(FALLBACK["chain_id"],FALLBACK["chain_name"],FALLBACK["rpc_urls"],to_checksum_address(FALLBACK["contract"]),"verified fallback")


def call(contract: str, selector: str) -> tuple[str,list[Any]]:
    return "eth_call",[{"to":contract,"data":selector},"latest"]


def as_int(raw: str) -> int:
    if not raw or raw=="0x":raise RpcError("contract returned empty data")
    return int(raw,16)


def state(rpc: RpcPool, contract: str, address: str) -> Job:
    supply,price,challenge,difficulty,tx_nonce,gas_price,block,balance,last_mine_block=rpc.batch([call(contract,TOTAL_SUPPLY),call(contract,MINT_PRICE),call(contract,CHALLENGE),call(contract,DIFFICULTY),("eth_getTransactionCount",[address,"pending"]),("eth_gasPrice",[]),("eth_getBlockByNumber",["latest",False]),("eth_getBalance",[address,"latest"]),("eth_getStorageAt",[contract,LAST_MINE_BLOCK_SLOT,"latest"])])
    if not re.fullmatch(r"0x[0-9a-fA-F]{64}",challenge or ""):raise RpcError("invalid challenge")
    rawbase=block.get("baseFeePerGas")
    return Job(as_int(supply),as_int(price),challenge.lower(),as_int(difficulty),as_int(tx_nonce),as_int(gas_price),int(rawbase,16) if rawbase is not None else None,as_int(balance),as_int(block["number"]),as_int(last_mine_block))


def fresh(rpc: RpcPool, contract: str) -> tuple[str,int]:
    challenge,difficulty=rpc.batch([call(contract,CHALLENGE),call(contract,DIFFICULTY)])
    return challenge.lower(),as_int(difficulty)


def monitor(rpc: RpcPool, contract: str, address: str) -> tuple[str,int,int,int,int,int|None,int,int,int]:
    challenge,difficulty,price,tx_nonce,gas_price,block,balance,last_mine_block=rpc.batch([call(contract,CHALLENGE),call(contract,DIFFICULTY),call(contract,MINT_PRICE),("eth_getTransactionCount",[address,"pending"]),("eth_gasPrice",[]),("eth_getBlockByNumber",["latest",False]),("eth_getBalance",[address,"latest"]),("eth_getStorageAt",[contract,LAST_MINE_BLOCK_SLOT,"latest"])])
    rawbase=block.get("baseFeePerGas")
    return challenge.lower(),as_int(difficulty),as_int(price),as_int(tx_nonce),as_int(gas_price),int(rawbase,16) if rawbase is not None else None,as_int(balance),as_int(block["number"]),as_int(last_mine_block)


def digest(address: str, nonce: int, challenge: str) -> bytes:
    payload=bytes.fromhex(address[2:])+nonce.to_bytes(32,"big")+bytes.fromhex(challenge[2:])
    return hashlib.sha256(payload).digest()


def zero_bits(value: bytes) -> int:
    result=0
    for byte in value:
        if byte==0:result+=8
        else:result+=8-byte.bit_length();break
    return result


def words(address: str, challenge: str) -> np.ndarray:
    raw=bytes.fromhex(address[2:])+bytes.fromhex(challenge[2:])
    return np.asarray([int.from_bytes(raw[i:i+4],"big") for i in range(0,52,4)],dtype=np.uint32)


class GPU:
    def __init__(self):
        props=cp.cuda.runtime.getDeviceProperties(0);raw=props.get("name",b"NVIDIA GPU")
        self.name=raw.decode(errors="replace") if isinstance(raw,bytes) else str(raw)
        self.blocks=max(64,int(props.get("multiProcessorCount",1))*8);self.iters=128
        module=cp.RawModule(code=CUDA_SOURCE,options=("--std=c++14","--use_fast_math"),name_expressions=("hashgoat_mine","hashgoat_hash_one"))
        self.kernel=module.get_function("hashgoat_mine");self.one=module.get_function("hashgoat_hash_one")
        self.found=cp.zeros(1,cp.uint32);self.answer=cp.zeros(1,cp.uint64);self.best=cp.zeros(1,cp.uint32);self.bestnonce=cp.zeros(1,cp.uint64)

    def test(self) -> None:
        address="0x"+"11"*20;challenge="0x"+"42"*32;base=cp.asarray(words(address,challenge));out=cp.zeros(8,cp.uint32)
        for nonce in (0,1,0x1122334455667788):
            self.one((1,),(1,),(base,np.uint32(nonce & 0xFFFFFFFF),np.uint32((nonce >> 32) & 0xFFFFFFFF),out));got=b"".join(int(x).to_bytes(4,"big") for x in cp.asnumpy(out))
            if got!=digest(address,nonce,challenge):raise RuntimeError("GPU SHA-256 self-test failed")
        # Successful on-chain mine from block 62795200: guards field order and encoding.
        address="0xb1B825a870caD9170A761C78357B2F945c5468a0";challenge="0xb9692537404968abba692b7cb029ac9e102fe619734ee05efb0d0b6b94bbc790";nonce=int("47a09f3d58f3f01fae742fc806f648b0b04e673dbb8e8659465ea88cfef5f06c",16)
        if zero_bits(digest(address,nonce,challenge))<30:raise RuntimeError("on-chain proof self-test failed")

    def mine(self,address: str,job: Job,rpc: RpcPool,contract: str,ui: "UI",session: int) -> tuple[str,int|None,int]:
        base=cp.asarray(words(address,job.challenge));start=secrets.randbits(64);offset=0;count=0;rate=0.;best=0;besthash="-";next_poll=0.;probe=None
        watcher=ThreadPoolExecutor(max_workers=1);watch_rpc=RpcPool(rpc.urls)
        try:
            while True:
                now=time.monotonic()
                if probe is None and now>=next_poll:
                    probe=watcher.submit(monitor,watch_rpc,contract,address);next_poll=now+POLL_SECONDS
                self.found.fill(0);self.answer.fill(0);self.best.fill(0);self.bestnonce.fill(0)
                first=(start+offset)&((1<<64)-1);hashes=self.blocks*THREADS*self.iters;began=time.perf_counter()
                self.kernel((self.blocks,),(THREADS,),(base,np.uint32(first & 0xFFFFFFFF),np.uint32((first >> 32) & 0xFFFFFFFF),np.uint32(self.iters),np.uint32(job.difficulty),self.found,self.answer,self.best,self.bestnonce));cp.cuda.Stream.null.synchronize();elapsed=max(time.perf_counter()-began,.001)
                hit=int(cp.asnumpy(self.found)[0]);nonce=int(cp.asnumpy(self.answer)[0]);batchbest=int(cp.asnumpy(self.best)[0]);bn=int(cp.asnumpy(self.bestnonce)[0]);instant=hashes/elapsed;rate=instant if not rate else rate*.72+instant*.28;count+=hashes;offset=(offset+hashes)&((1<<64)-1)
                if batchbest>best:
                    value=digest(address,bn,job.challenge);bits=zero_bits(value)
                    if bits>=best:best=bits;besthash="0x"+value.hex()
                ui.data.update(phase="MINING",rate=rate,job_hashes=count,session_hashes=session+count,best=best,besthash=besthash,batch_ms=elapsed*1000,iters=self.iters);ui.refresh()
                factor=min(1.45,max(.70,TARGET_BATCH_SECONDS/elapsed));self.iters=max(16,min(16384,int(self.iters*factor)))
                if probe is not None and probe.done():
                    try:
                        challenge,difficulty,price,tx_nonce,gas_price,base_fee,balance,latest_block,last_mine_block=probe.result()
                        if challenge!=job.challenge or difficulty!=job.difficulty or price!=job.price:return "stale",None,count
                        job.tx_nonce=tx_nonce;job.gas_price=gas_price;job.base_fee=base_fee;job.balance=balance
                        job.latest_block=latest_block;job.last_mine_block=last_mine_block
                    except Exception as exc:ui.log(f"RPC monitor retry: {exc}","yellow")
                    probe=None
                if hit:
                    if zero_bits(digest(address,nonce,job.challenge))<job.difficulty:raise RuntimeError("GPU nonce failed CPU verification")
                    return "found",nonce,count
        finally:
            watcher.shutdown(wait=False,cancel_futures=True)


class UI:
    def __init__(self,config: Config,address: str,gpu: str):
        self.config=config;self.address=address;self.gpu=gpu;self.logs:deque[tuple[str,str]]=deque(maxlen=6);self.live:Live|None=None
        self.data={"phase":"STARTING","supply":0,"difficulty":0,"price":0,"challenge":"-","rate":0.,"job_hashes":0,"session_hashes":0,"best":0,"besthash":"-","batch_ms":0.,"iters":0,"mints":0,"tx":"-"}
    @staticmethod
    def short(v: str)->str:return v if len(v)<35 else v[:18]+"..."+v[-12:]
    @staticmethod
    def rate(v: float)->str:
        if v>=1e9:return f"{v/1e9:.2f} GH/s"
        if v>=1e6:return f"{v/1e6:.2f} MH/s"
        return f"{v/1e3:.2f} KH/s"
    @staticmethod
    def count(v: int)->str:
        if v>=10**9:return f"{v/1e9:.2f}B"
        if v>=10**6:return f"{v/1e6:.2f}M"
        return f"{v:,}"
    def log(self,message: str,color: str="cyan")->None:self.logs.append((time.strftime("%H:%M:%S  ")+message,color));self.refresh()
    def render(self)->Group:
        d=self.data;top=Table.grid(expand=True);top.add_column(style="bold cyan",width=13);top.add_column();top.add_column(style="bold cyan",width=12);top.add_column()
        top.add_row("NETWORK",f"{self.config.chain_name} ({self.config.chain_id})","PHASE",str(d["phase"]));top.add_row("WALLET",self.short(self.address),"GPU",self.gpu);top.add_row("CONTRACT",self.short(self.config.contract),"PRICE","FREE" if d["price"]==0 else f"{d['price']/1e18:.8f} ETH");top.add_row("SUPPLY",f"{d['supply']:,} / 10,000","TARGET",f"{d['difficulty']} bits");top.add_row("RPC ROUTES",str(len(self.config.rpc_urls)),"MODE","FREE / CAPPED PAID")
        mining=Table.grid(expand=True);mining.add_column(style="bright_green",width=15);mining.add_column();mining.add_row("HASHRATE",self.rate(float(d["rate"])));mining.add_row("BEST",f"{d['best']} / {d['difficulty']} bits");mining.add_row("HASHES",f"job {self.count(d['job_hashes'])} | session {self.count(d['session_hashes'])}");mining.add_row("GPU BATCH",f"{d['batch_ms']:.0f} ms | {d['iters']} iterations/thread");mining.add_row("DIFFICULTY",f"live target {d['difficulty']} bits");mining.add_row("CHALLENGE",self.short(str(d["challenge"])));mining.add_row("BEST HASH",self.short(str(d["besthash"])));mining.add_row("MINTED",str(d["mints"]));mining.add_row("LAST TX",self.short(str(d["tx"])))
        lines=[Text(x,style=c) for x,c in self.logs] or [Text("Starting...",style="dim")]
        return Group(Panel(top,title="HASHGOAT GPU AUTO-MINER",border_style="bright_cyan"),Panel(mining,title="LIVE MINING",border_style="bright_green"),Panel(Group(*lines),title="ACTIVITY",border_style="blue"),Text(" Ctrl+C: stop safely | private key is memory-only ",style="bold black on bright_cyan"))
    def refresh(self)->None:
        if self.live:self.live.update(self.render(),refresh=True)


def calldata(nonce: int,challenge: str)->str:return MINE+nonce.to_bytes(32,"big").hex()+challenge[2:]


def gas_bump(value: int) -> int:
    return max(value + 1, value * GAS_BUMP_NUM // GAS_BUMP_DEN)


def wait_open_block(rpc: RpcPool, config: Config, job: Job, ui: "UI") -> None:
    # The monitor already keeps this guard hot. An extra RPC check on every hit
    # used to add hundreds of milliseconds to the critical submission path.
    if job.last_mine_block!=job.latest_block:return
    while True:
        block_hex,last_block_hex,challenge=rpc.batch([
            ("eth_blockNumber",[]),
            ("eth_getStorageAt",[config.contract,LAST_MINE_BLOCK_SLOT,"latest"]),
            call(config.contract,CHALLENGE),
        ])
        if challenge.lower()!=job.challenge:raise RpcError("STALE: challenge changed before submission")
        if as_int(last_block_hex)!=as_int(block_hex):return
        ui.data["phase"]="WAIT BLOCK";ui.log("Same-block guard active; waiting next block with 0 gas","yellow");ui.refresh()
        time.sleep(.15)


def submit(rpc: RpcPool,config: Config,account: Any,nonce: int,job: Job,ui: "UI")->str:
    wait_open_block(rpc,config,job,ui)
    gasprice=job.gas_price
    if job.base_fee is None:
        unit=max(gas_bump(gasprice),MIN_PRIORITY_FEE);fee={"gasPrice":unit}
    else:
        tip=max(gasprice-job.base_fee,gasprice,MIN_PRIORITY_FEE)
        unit=max(gas_bump(gasprice),job.base_fee*2+tip);fee={"type":2,"maxFeePerGas":unit,"maxPriorityFeePerGas":tip}
    required=job.price+MINE_GAS_LIMIT*unit
    if job.balance<required:raise RuntimeError(f"insufficient ETH; need up to {required/1e18:.9f} ETH")
    tx={"chainId":config.chain_id,"nonce":job.tx_nonce,"to":config.contract,"value":job.price,"data":calldata(nonce,job.challenge),"gas":MINE_GAS_LIMIT,**fee}
    signed=account.sign_transaction(tx);raw=getattr(signed,"raw_transaction",None) or signed.rawTransaction;raw_hex="0x"+bytes(raw).hex();tx_hash="0x"+bytes(signed.hash).hex()
    return rpc.broadcast(raw_hex,tx_hash)


def receipt(rpc: RpcPool,txhash: str)->dict[str,Any]:
    until=time.monotonic()+180
    while time.monotonic()<until:
        value=rpc.call("eth_getTransactionReceipt",[txhash])
        if value:return value
        time.sleep(.4)
    raise TimeoutError("receipt timeout")


def revert_reason(rpc: RpcPool, contract: str, job: Job) -> str:
    try:
        challenge,difficulty=fresh(rpc,contract)
        if challenge!=job.challenge:return "Race lost: another miner changed the challenge first"
        if difficulty!=job.difficulty:return "Stale: difficulty changed before inclusion"
        return "Reverted with same challenge; refreshing job"
    except Exception:
        return "Transaction reverted; refreshing"


def private_account()->Any:
    key=getpass.getpass("PRIVATE_KEY (hidden, never saved): ").strip()
    if not key:raise ValueError("empty private key")
    if not key.startswith("0x"):key="0x"+key
    try:return Account.from_key(key)
    finally:key=""


def runtime_options()->tuple[list[str],Limits]:
    raw_urls=getpass.getpass("PREMIUM HTTP RPC URL(S) (hidden, optional): ").strip()
    urls=[]
    if raw_urls:
        for url in re.split(r"[,\s]+",raw_urls):
            if not re.match(r"^https?://",url,re.I):raise ValueError("premium RPC must start with http:// or https://")
            urls.append(url.rstrip("/"))
    raw_price=input("MAX MINT PRICE ETH (0 = FREE only): ").strip() or "0"
    try:
        maximum=Decimal(raw_price)
        if maximum<0:raise ValueError
        max_price=int(maximum*Decimal(10**18))
    except (InvalidOperation,ValueError):raise ValueError("invalid maximum mint price")
    raw_count=input("MAX PAID MINTS (Enter = 1, 0 = unlimited): ").strip() or "1"
    try:max_paid=int(raw_count)
    except ValueError:raise ValueError("invalid paid mint limit")
    if max_paid<0:raise ValueError("paid mint limit cannot be negative")
    return urls,Limits(max_price,max_paid)


def run()->int:
    console=Console();console.print("[bold cyan]HashGoat GPU Auto-Miner[/bold cyan]")
    try:account=private_account()
    except Exception as exc:console.print(f"[red]Invalid private key: {exc}[/red]");return 2
    try:premium,limits=runtime_options()
    except Exception as exc:console.print(f"[red]Invalid runtime option: {exc}[/red]");return 2
    discovered=discover();config=Config(discovered.chain_id,discovered.chain_name,list(dict.fromkeys(premium+discovered.rpc_urls)),discovered.contract,discovered.source);rpc=RpcPool(config.rpc_urls)
    try:
        warmed=rpc.warmup(config.chain_id)
        config=Config(config.chain_id,config.chain_name,rpc.urls,config.contract,config.source)
        if rpc.call("eth_getCode",[config.contract,"latest"]) in ("0x",None):raise RuntimeError("contract bytecode missing")
        gpu=GPU();gpu.test()
    except Exception as exc:console.print(f"[red]Startup failed: {exc}[/red]");return 1
    ui=UI(config,account.address,gpu.name);session=0;paid_mints=0
    with Live(ui.render(),console=console,refresh_per_second=4,screen=False) as live:
        ui.live=live;ui.log("GPU SHA-256 self-test passed","green");ui.log("RPC warm: "+" | ".join(f"#{i+1} {ms:.0f}ms" for i,(_,ms) in enumerate(warmed)),"green");ui.log(f"{config.source}; paid price cap {limits.max_price/1e18:.8f} ETH")
        try:
            while True:
                job=state(rpc,config.contract,account.address);ui.data.update(phase="READY",supply=job.supply,difficulty=job.difficulty,price=job.price,challenge=job.challenge,job_hashes=0,best=0,besthash="-");ui.refresh()
                if job.price>limits.max_price:raise RuntimeError(f"PRICE GUARD: {job.price/1e18:.8f} ETH exceeds cap {limits.max_price/1e18:.8f} ETH")
                if job.price>0 and limits.max_paid_mints and paid_mints>=limits.max_paid_mints:ui.data["phase"]="PAID LIMIT";ui.log("Paid mint limit reached; stopped before spending more","yellow");break
                if job.supply>=10000:ui.data["phase"]="SOLD OUT";ui.log("Collection sold out","yellow");break
                ui.log(f"Mining fresh challenge at {job.difficulty} bits","bright_green");status,nonce,used=gpu.mine(account.address,job,rpc,config.contract,ui,session);session+=used;ui.data["session_hashes"]=session
                if status=="stale":ui.log("Challenge changed; switched jobs with 0 gas","yellow");continue
                ui.data["phase"]="SUBMITTING";ui.log(f"Valid {job.difficulty}-bit proof; immediate {len(rpc.urls)}-route broadcast","green");ui.refresh()
                try:txhash=submit(rpc,config,account,int(nonce),job,ui)
                except RpcError as exc:ui.log(str(exc),"yellow");continue
                ui.data.update(phase="CONFIRMING",tx=txhash);ui.log(f"Submitted {ui.short(txhash)}");result=receipt(rpc,txhash);usedgas=int(result.get("gasUsed","0x0"),16)
                if int(result.get("status","0x0"),16)==1:
                    ui.data["mints"]+=1;paid_mints+=int(job.price>0);ui.data["phase"]="MINTED";ui.log(f"MINT SUCCESS #{ui.data['mints']} | gas {usedgas:,}","bold green")
                else:ui.data["phase"]="REVERTED";ui.log(revert_reason(rpc,config.contract,job),"red")
                ui.refresh();time.sleep(.25)
        except KeyboardInterrupt:ui.data["phase"]="STOPPED";ui.log("Stopped safely by user","yellow");ui.refresh()
        except Exception as exc:ui.data["phase"]="ERROR";ui.log(str(exc),"red");ui.refresh();return 1
        finally:ui.live=None
    return 0


if __name__=="__main__":sys.exit(run())
