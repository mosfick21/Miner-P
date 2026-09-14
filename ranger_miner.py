#!/usr/bin/env python3
"""Hash Rangers CUDA auto-miner. Secrets are prompted and never saved."""

from __future__ import annotations

import getpass
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
from eth_utils import keccak, to_checksum_address
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CHAIN_ID = 4663
CHAIN_NAME = "Robinhood Chain"
CONTRACT = to_checksum_address("0xccC0A6184EE66389448b81C2c96353615bC6E9fC")
RPC_URLS = ["https://rpc.mainnet.chain.robinhood.com", "https://robinhood.drpc.org"]
MAX_SUPPLY = 2222
BLOCK_HASH_OFFSET = 2

TOTAL_MINTED = "0xa2309ff8"
REMAINING = "0xc1747588"
CURRENT_PRICE = "0x9d1b464a"
CURRENT_TARGET = "0x39148c53"
LAST_WORK_HASH = "0x4fd3b2bd"
CURRENT_BLOCK = "0x378ec23b"
BLOCK_HASH_OF = "0xfad1919a"
PAUSED = "0x5c975abb"
MINE = "0x071e9503"  # mine(uint256,uint256)

POLL_SECONDS = 0.18
TARGET_BATCH_SECONDS = 0.18
THREADS = 256
MINE_GAS_LIMIT = 300_000
MIN_PRIORITY_FEE = 20_000_000

CUDA_SOURCE = r"""
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

__device__ __constant__ uint64_t RC[24]={
0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};
__device__ __constant__ int ROT[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
__device__ __constant__ int PIL[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};

__device__ __forceinline__ uint32_t sw32(uint32_t x){return __byte_perm(x,0,0x0123);}
__device__ __forceinline__ uint2 xo(uint2 a,uint2 b){return make_uint2(a.x^b.x,a.y^b.y);}
__device__ __forceinline__ uint2 an(uint2 a,uint2 b){return make_uint2(a.x&b.x,a.y&b.y);}
__device__ __forceinline__ uint2 nt(uint2 a){return make_uint2(~a.x,~a.y);}
__device__ __forceinline__ uint2 rol(uint2 v,int n){
 if(n==0)return v;
 if(n<32)return make_uint2((v.x<<n)|(v.y>>(32-n)),(v.y<<n)|(v.x>>(32-n)));
 if(n==32)return make_uint2(v.y,v.x);
 n-=32;return make_uint2((v.y<<n)|(v.x>>(32-n)),(v.x<<n)|(v.y>>(32-n)));
}
__device__ __forceinline__ void keccakf(uint2 s[25]){
 uint2 bc[5],t;
 #pragma unroll
 for(int r=0;r<24;r++){
  #pragma unroll
  for(int i=0;i<5;i++)bc[i]=xo(xo(xo(xo(s[i],s[i+5]),s[i+10]),s[i+15]),s[i+20]);
  #pragma unroll
  for(int i=0;i<5;i++){t=xo(bc[(i+4)%5],rol(bc[(i+1)%5],1));for(int j=0;j<25;j+=5)s[j+i]=xo(s[j+i],t);}
  t=s[1];
  #pragma unroll
  for(int i=0;i<24;i++){int j=PIL[i];bc[0]=s[j];s[j]=rol(t,ROT[i]);t=bc[0];}
  #pragma unroll
  for(int j=0;j<25;j+=5){for(int i=0;i<5;i++)bc[i]=s[j+i];for(int i=0;i<5;i++)s[j+i]=xo(bc[i],an(nt(bc[(i+1)%5]),bc[(i+2)%5]));}
  uint64_t rc=RC[r];s[0].x^=(uint32_t)rc;s[0].y^=(uint32_t)(rc>>32);
 }
}
__device__ __forceinline__ void hash_nonce(const uint32_t*base,uint32_t lo,uint32_t hi,uint32_t out[8]){
 uint2 s[25];
 #pragma unroll
 for(int i=0;i<25;i++)s[i]=i<17?make_uint2(base[2*i],base[2*i+1]):make_uint2(0u,0u);
 s[5]=make_uint2(0u,sw32(hi));
 s[6].x=sw32(lo);
 keccakf(s);
 #pragma unroll
 for(int i=0;i<4;i++){out[2*i]=sw32(s[i].x);out[2*i+1]=sw32(s[i].y);}
}
__device__ __forceinline__ uint32_t zero_bits(const uint32_t h[8]){uint32_t n=0;for(int i=0;i<8;i++){if(h[i]==0)n+=32;else{n+=__clz(h[i]);break;}}return n;}
__device__ __forceinline__ bool below(const uint32_t h[8],const uint32_t*t){for(int i=0;i<8;i++){if(h[i]<t[i])return true;if(h[i]>t[i])return false;}return false;}

extern "C" __global__ void ranger_mine(const uint32_t*base,const uint32_t*target,uint32_t slo,uint32_t shi,uint32_t iters,uint32_t*found,uint64_t*answer,uint32_t*best,uint64_t*bestnonce){
 uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x,stride=(uint64_t)gridDim.x*blockDim.x,start=((uint64_t)shi<<32)|slo;
 uint32_t local=0;uint64_t localnonce=start+tid;
 for(uint32_t i=0;i<iters;i++){if(__ldg(found))break;uint64_t nonce=start+tid+(uint64_t)i*stride;uint32_t h[8];hash_nonce(base,(uint32_t)nonce,(uint32_t)(nonce>>32),h);uint32_t z=zero_bits(h);if(z>local){local=z;localnonce=nonce;}if(below(h,target)){if(atomicCAS(found,0u,1u)==0u)*answer=nonce;break;}}
 uint32_t old=atomicMax(best,local);if(local>old)*bestnonce=localnonce;
}
extern "C" __global__ void ranger_hash_one(const uint32_t*base,uint32_t lo,uint32_t hi,uint32_t*out){if(blockIdx.x||threadIdx.x)return;uint32_t h[8];hash_nonce(base,lo,hi,h);for(int i=0;i<8;i++)out[i]=h[i];}
"""


class RpcError(RuntimeError):
    pass


class RpcPool:
    def __init__(self, urls: list[str]):
        self.urls=list(dict.fromkeys(urls));self.index=0;self.ident=0
        self.sessions={u:requests.Session() for u in self.urls}
        self.send_sessions={u:requests.Session() for u in self.urls}
        self.pool=ThreadPoolExecutor(max_workers=max(1,len(self.urls)))

    @staticmethod
    def request(session: requests.Session,url: str,method: str,params: list[Any]|None=None,timeout: float=12)->Any:
        response=session.post(url,json={"jsonrpc":"2.0","id":int(time.time_ns()%2_000_000_000),"method":method,"params":params or []},timeout=timeout)
        response.raise_for_status();body=response.json()
        if body.get("error"):raise RpcError(body["error"].get("message",str(body["error"])))
        return body["result"]

    def warmup(self)->list[float]:
        def probe(url: str)->tuple[str,float]:
            began=time.perf_counter();value=self.request(self.send_sessions[url],url,"eth_chainId",timeout=8)
            if int(value,16)!=CHAIN_ID:raise RpcError(f"wrong chain {int(value,16)}")
            return url,(time.perf_counter()-began)*1000
        good=[]
        futures={self.pool.submit(probe,u):u for u in self.urls}
        for future in as_completed(futures):
            try:good.append(future.result())
            except Exception:pass
        live={u for u,_ in good};self.urls=[u for u in self.urls if u in live]
        if not self.urls:raise RpcError("all RPC routes failed")
        return [dict(good)[u] for u in self.urls]

    def call(self,method: str,params: list[Any]|None=None)->Any:
        error=None
        for attempt in range(max(3,len(self.urls)*2)):
            url=self.urls[(self.index+attempt)%len(self.urls)]
            try:
                value=self.request(self.sessions[url],url,method,params);self.index=self.urls.index(url);return value
            except Exception as exc:error=exc
        raise RpcError(str(error))

    def batch(self,calls: list[tuple[str,list[Any]]])->list[Any]:
        error=None
        for attempt in range(max(3,len(self.urls)*2)):
            url=self.urls[(self.index+attempt)%len(self.urls)];payload=[];ids=[]
            for method,params in calls:
                self.ident+=1;ids.append(self.ident);payload.append({"jsonrpc":"2.0","id":self.ident,"method":method,"params":params})
            try:
                response=self.sessions[url].post(url,json=payload,timeout=12);response.raise_for_status();items=response.json();byid={x["id"]:x for x in items};out=[]
                for ident in ids:
                    item=byid[ident]
                    if item.get("error"):raise RpcError(item["error"].get("message",str(item["error"])))
                    out.append(item["result"])
                self.index=self.urls.index(url);return out
            except Exception as exc:error=exc
        # Some public providers reject JSON-RPC batches while accepting the
        # exact same calls individually.
        try:return [self.call(method,params) for method,params in calls]
        except Exception as exc:raise RpcError(f"batch/sequential RPC failed: {error} | {exc}")

    def broadcast(self,raw: str,expected: str)->str:
        def send(url: str)->tuple[bool,str]:
            try:return True,self.request(self.send_sessions[url],url,"eth_sendRawTransaction",[raw])
            except Exception as exc:return False,str(exc)
        errors=[];futures=[self.pool.submit(send,u) for u in self.urls]
        for future in as_completed(futures):
            ok,value=future.result()
            if ok:return value
            errors.append(value)
        if any("already known" in x.lower() or "nonce too low" in x.lower() for x in errors):return expected
        raise RpcError("broadcast failed: "+" | ".join(errors))


@dataclass
class Job:
    total: int;remaining: int;price: int;target: int;work: str;anchor: int;block_hash: str;paused: bool
    tx_nonce: int;gas_price: int;base_fee: int|None;balance: int;latest_block: int


def c(selector: str,args: str="")->tuple[str,list[Any]]:
    return "eth_call",[{"to":CONTRACT,"data":selector+args},"latest"]


def n(raw: str)->int:
    if not raw or raw=="0x":raise RpcError("empty contract response")
    return int(raw,16)


def word(value: int)->str:
    return value.to_bytes(32,"big").hex()


def get_job(rpc: RpcPool,address: str)->Job:
    total,remaining,price,target,work,current,paused,tx_nonce,gas_price,block,balance=rpc.batch([
        c(TOTAL_MINTED),c(REMAINING),c(CURRENT_PRICE),c(CURRENT_TARGET),c(LAST_WORK_HASH),c(CURRENT_BLOCK),c(PAUSED),
        ("eth_getTransactionCount",[address,"pending"]),("eth_gasPrice",[]),("eth_getBlockByNumber",["latest",False]),("eth_getBalance",[address,"latest"]),
    ])
    anchor=max(0,n(current)-BLOCK_HASH_OFFSET);block_hash=rpc.call(*c(BLOCK_HASH_OF,word(anchor)))
    if not re.fullmatch(r"0x[0-9a-fA-F]{64}",work or "") or not re.fullmatch(r"0x[0-9a-fA-F]{64}",block_hash or ""):raise RpcError("invalid mining state")
    base=block.get("baseFeePerGas")
    return Job(n(total),n(remaining),n(price),n(target),work.lower(),anchor,block_hash.lower(),bool(n(paused)),n(tx_nonce),n(gas_price),int(base,16) if base else None,n(balance),n(block["number"]))


def monitor(rpc: RpcPool,address: str)->tuple[str,int,int,int,int,int|None,int,int]:
    work,target,price,current,tx_nonce,gas_price,block,balance=rpc.batch([
        c(LAST_WORK_HASH),c(CURRENT_TARGET),c(CURRENT_PRICE),c(CURRENT_BLOCK),("eth_getTransactionCount",[address,"pending"]),("eth_gasPrice",[]),("eth_getBlockByNumber",["latest",False]),("eth_getBalance",[address,"latest"]),
    ])
    base=block.get("baseFeePerGas")
    return work.lower(),n(target),n(price),n(current),n(tx_nonce),n(gas_price),int(base,16) if base else None,n(balance)


def payload(address: str,nonce: int,job: Job)->bytes:
    return bytes.fromhex(address[2:])+nonce.to_bytes(32,"big")+bytes.fromhex(job.work[2:])+bytes.fromhex(job.block_hash[2:])


def digest(address: str,nonce: int,job: Job)->bytes:
    return keccak(payload(address,nonce,job))


def bits(value: bytes)->int:
    count=0
    for byte in value:
        if byte==0:count+=8
        else:count+=8-byte.bit_length();break
    return count


def base_lanes(address: str,job: Job)->np.ndarray:
    raw=bytearray(136);raw[:20]=bytes.fromhex(address[2:]);raw[52:84]=bytes.fromhex(job.work[2:]);raw[84:116]=bytes.fromhex(job.block_hash[2:]);raw[116]^=1;raw[135]^=128
    return np.asarray([int.from_bytes(raw[i:i+4],"little") for i in range(0,136,4)],dtype=np.uint32)


def target_words(target: int)->np.ndarray:
    raw=target.to_bytes(32,"big");return np.asarray([int.from_bytes(raw[i:i+4],"big") for i in range(0,32,4)],dtype=np.uint32)


class GPU:
    def __init__(self):
        props=cp.cuda.runtime.getDeviceProperties(0);raw=props.get("name",b"NVIDIA GPU");self.name=raw.decode(errors="replace") if isinstance(raw,bytes) else str(raw)
        self.blocks=max(64,int(props.get("multiProcessorCount",1))*8);self.iters=128
        module=cp.RawModule(code=CUDA_SOURCE,options=("--std=c++14","--use_fast_math"),name_expressions=("ranger_mine","ranger_hash_one"))
        self.kernel=module.get_function("ranger_mine");self.one=module.get_function("ranger_hash_one")
        self.found=cp.zeros(1,cp.uint32);self.answer=cp.zeros(1,cp.uint64);self.best=cp.zeros(1,cp.uint32);self.bestnonce=cp.zeros(1,cp.uint64)

    def test(self,address: str,job: Job)->None:
        base=cp.asarray(base_lanes(address,job));out=cp.zeros(8,cp.uint32)
        for nonce in (0,1,0x1122334455667788):
            self.one((1,),(1,),(base,np.uint32(nonce&0xffffffff),np.uint32(nonce>>32),out));got=b"".join(int(x).to_bytes(4,"big") for x in cp.asnumpy(out))
            if got!=digest(address,nonce,job):raise RuntimeError("GPU Keccak self-test failed")

    def mine(self,address: str,job: Job,rpc: RpcPool,ui: "UI",session: int)->tuple[str,int|None,int]:
        base=cp.asarray(base_lanes(address,job));target=cp.asarray(target_words(job.target));start=secrets.randbits(64);offset=0;count=0;rate=0.;best=0;besthash="-";probe=None;next_poll=0.
        watcher=ThreadPoolExecutor(max_workers=1);watch_rpc=RpcPool(rpc.urls)
        try:
            while True:
                now=time.monotonic()
                if probe is None and now>=next_poll:probe=watcher.submit(monitor,watch_rpc,address);next_poll=now+POLL_SECONDS
                self.found.fill(0);self.answer.fill(0);self.best.fill(0);self.bestnonce.fill(0)
                first=(start+offset)&((1<<64)-1);hashes=self.blocks*THREADS*self.iters;began=time.perf_counter()
                self.kernel((self.blocks,),(THREADS,),(base,target,np.uint32(first&0xffffffff),np.uint32(first>>32),np.uint32(self.iters),self.found,self.answer,self.best,self.bestnonce));cp.cuda.Stream.null.synchronize();elapsed=max(time.perf_counter()-began,.001)
                hit=int(cp.asnumpy(self.found)[0]);nonce=int(cp.asnumpy(self.answer)[0]);batchbest=int(cp.asnumpy(self.best)[0]);bestnonce=int(cp.asnumpy(self.bestnonce)[0]);instant=hashes/elapsed;rate=instant if not rate else rate*.72+instant*.28;count+=hashes;offset=(offset+hashes)&((1<<64)-1)
                if batchbest>best:best=batchbest;besthash="0x"+digest(address,bestnonce,job).hex()
                ui.data.update(phase="MINING",rate=rate,job_hashes=count,session_hashes=session+count,best=best,besthash=besthash,batch_ms=elapsed*1000,iters=self.iters);ui.refresh()
                self.iters=max(16,min(16384,int(self.iters*min(1.45,max(.70,TARGET_BATCH_SECONDS/elapsed)))))
                if probe is not None and probe.done():
                    try:
                        work,newtarget,price,current,tx_nonce,gas_price,base_fee,balance=probe.result()
                        if work!=job.work or newtarget!=job.target or price!=job.price or current-job.anchor>=200:return "stale",None,count
                        job.tx_nonce=tx_nonce;job.gas_price=gas_price;job.base_fee=base_fee;job.balance=balance
                    except Exception as exc:ui.log(f"RPC monitor retry: {exc}","yellow")
                    probe=None
                if hit:
                    proof=digest(address,nonce,job)
                    if int.from_bytes(proof,"big")>=job.target:raise RuntimeError("GPU nonce failed CPU verification")
                    return "found",nonce,count
        finally:watcher.shutdown(wait=False,cancel_futures=True)


class UI:
    def __init__(self,address: str,gpu: str,routes: int):
        self.address=address;self.gpu=gpu;self.routes=routes;self.logs:deque[tuple[str,str]]=deque(maxlen=6);self.live:Live|None=None
        self.data={"phase":"STARTING","total":0,"price":0,"target":0,"rate":0.,"job_hashes":0,"session_hashes":0,"best":0,"besthash":"-","batch_ms":0.,"iters":0,"mints":0,"tx":"-"}
    @staticmethod
    def short(v: str)->str:return v if len(v)<35 else v[:18]+"..."+v[-12:]
    @staticmethod
    def rate(v: float)->str:return f"{v/1e9:.2f} GH/s" if v>=1e9 else f"{v/1e6:.2f} MH/s"
    @staticmethod
    def count(v: int)->str:return f"{v/1e9:.2f}B" if v>=10**9 else f"{v/1e6:.2f}M" if v>=10**6 else f"{v:,}"
    def log(self,msg: str,color: str="cyan")->None:self.logs.append((time.strftime("%H:%M:%S  ")+msg,color));self.refresh()
    def render(self)->Group:
        d=self.data;top=Table.grid(expand=True);top.add_column(style="bold cyan",width=12);top.add_column();top.add_column(style="bold cyan",width=12);top.add_column()
        top.add_row("NETWORK",f"{CHAIN_NAME} ({CHAIN_ID})","PHASE",d["phase"]);top.add_row("WALLET",self.short(self.address),"GPU",self.gpu);top.add_row("CONTRACT",self.short(CONTRACT),"PRICE","FREE" if d["price"]==0 else f"{d['price']/1e18:.6f} ETH");top.add_row("SUPPLY",f"{d['total']:,} / {MAX_SUPPLY:,}","RPC ROUTES",str(self.routes))
        mine=Table.grid(expand=True);mine.add_column(style="bright_green",width=14);mine.add_column();mine.add_row("HASHRATE",self.rate(d["rate"]));mine.add_row("BEST",f"{d['best']} leading-zero bits");mine.add_row("HASHES",f"job {self.count(d['job_hashes'])} | session {self.count(d['session_hashes'])}");mine.add_row("GPU BATCH",f"{d['batch_ms']:.0f} ms | {d['iters']} iterations/thread");mine.add_row("TARGET",f"0x{d['target']:064x}" if d["target"] else "-");mine.add_row("BEST HASH",self.short(d["besthash"]));mine.add_row("MINTED",str(d["mints"]));mine.add_row("LAST TX",self.short(d["tx"]))
        lines=[Text(x,style=color) for x,color in self.logs] or [Text("Starting...",style="dim")]
        return Group(Panel(top,title="HASH RANGERS GPU AUTO-MINER",border_style="bright_cyan"),Panel(mine,title="LIVE MINING",border_style="bright_green"),Panel(Group(*lines),title="ACTIVITY",border_style="blue"),Text(" Ctrl+C: stop safely | secrets are memory-only ",style="bold black on bright_cyan"))
    def refresh(self)->None:
        if self.live:self.live.update(self.render(),refresh=True)


def calldata(nonce: int,anchor: int)->str:
    return MINE+word(nonce)+word(anchor)


def submit(rpc: RpcPool,account: Any,nonce: int,job: Job)->str:
    gasprice=job.gas_price
    if job.base_fee is None:unit=max(gasprice*125//100,MIN_PRIORITY_FEE);fees={"gasPrice":unit}
    else:
        tip=max(gasprice,MIN_PRIORITY_FEE);unit=max(gasprice*125//100,job.base_fee*2+tip);fees={"type":2,"maxFeePerGas":unit,"maxPriorityFeePerGas":tip}
    need=job.price+MINE_GAS_LIMIT*unit
    if job.balance<need:raise RuntimeError(f"insufficient ETH; need up to {need/1e18:.8f} ETH")
    tx={"chainId":CHAIN_ID,"nonce":job.tx_nonce,"to":CONTRACT,"value":job.price,"data":calldata(nonce,job.anchor),"gas":MINE_GAS_LIMIT,**fees}
    signed=account.sign_transaction(tx);raw=getattr(signed,"raw_transaction",None) or signed.rawTransaction
    return rpc.broadcast("0x"+bytes(raw).hex(),"0x"+bytes(signed.hash).hex())


def receipt(rpc: RpcPool,txhash: str)->dict[str,Any]:
    until=time.monotonic()+180
    while time.monotonic()<until:
        result=rpc.call("eth_getTransactionReceipt",[txhash])
        if result:return result
        time.sleep(.35)
    raise TimeoutError("receipt timeout")


def options()->tuple[Any,list[str],int,int]:
    key=getpass.getpass("PRIVATE_KEY (hidden, never saved): ").strip()
    if not key:raise ValueError("empty private key")
    if not key.startswith("0x"):key="0x"+key
    try:account=Account.from_key(key)
    finally:key=""
    premium=getpass.getpass("PREMIUM HTTP RPC URL(S) (hidden, optional): ").strip();urls=[]
    if premium:
        for url in re.split(r"[,\s]+",premium):
            if not re.match(r"^https?://",url,re.I):raise ValueError("RPC must start with http:// or https://")
            urls.append(url.rstrip("/"))
    raw=input("MAX MINT PRICE ETH (0 = FREE only): ").strip() or "0"
    try:max_price=int(Decimal(raw)*Decimal(10**18))
    except (InvalidOperation,ValueError):raise ValueError("invalid maximum price")
    raw_count=input("MAX PAID MINTS (Enter = 1, 0 = unlimited): ").strip() or "1"
    max_paid=int(raw_count)
    if max_price<0 or max_paid<0:raise ValueError("limits cannot be negative")
    return account,list(dict.fromkeys(urls+RPC_URLS)),max_price,max_paid


def run()->int:
    console=Console();console.print("[bold cyan]Hash Rangers GPU Auto-Miner[/bold cyan]")
    try:account,urls,max_price,max_paid=options();rpc=RpcPool(urls);warm=rpc.warmup();job=get_job(rpc,account.address);gpu=GPU();gpu.test(account.address,job)
    except Exception as exc:console.print(f"[red]Startup failed: {exc}[/red]");return 1
    ui=UI(account.address,gpu.name,len(rpc.urls));session=0;paid_mints=0
    with Live(ui.render(),console=console,refresh_per_second=4,screen=False) as live:
        ui.live=live;ui.log("GPU Keccak self-test passed","green");ui.log("RPC warm: "+" | ".join(f"#{i+1} {ms:.0f}ms" for i,ms in enumerate(warm)),"green")
        try:
            while True:
                job=get_job(rpc,account.address);ui.data.update(phase="READY",total=job.total,price=job.price,target=job.target,job_hashes=0,best=0,besthash="-");ui.refresh()
                if job.paused:raise RuntimeError("contract is paused")
                if job.remaining<=0 or job.total>=MAX_SUPPLY:ui.log("Collection sold out","yellow");break
                if job.price>max_price:raise RuntimeError(f"PRICE GUARD: {job.price/1e18:.6f} ETH exceeds cap {max_price/1e18:.6f} ETH")
                if job.price and max_paid and paid_mints>=max_paid:ui.log("Paid mint limit reached","yellow");break
                ui.log("Mining current proof","bright_green");status,nonce,used=gpu.mine(account.address,job,rpc,ui,session);session+=used;ui.data["session_hashes"]=session
                if status=="stale":ui.log("Mining state changed; switched with 0 gas","yellow");continue
                ui.data["phase"]="SUBMITTING";ui.log(f"Valid proof; immediate {len(rpc.urls)}-route broadcast","green");ui.refresh()
                txhash=submit(rpc,account,int(nonce),job);ui.data.update(phase="CONFIRMING",tx=txhash);ui.log(f"Submitted {ui.short(txhash)}")
                result=receipt(rpc,txhash);gas=n(result.get("gasUsed","0x0"))
                if n(result.get("status","0x0"))==1:ui.data["mints"]+=1;paid_mints+=int(job.price>0);ui.log(f"MINT SUCCESS #{ui.data['mints']} | gas {gas:,}","bold green")
                else:ui.log("Transaction reverted; proof/state lost the race","red")
                ui.refresh()
        except KeyboardInterrupt:ui.log("Stopped safely by user","yellow")
        except Exception as exc:ui.data["phase"]="ERROR";ui.log(str(exc),"red");return 1
        finally:ui.live=None
    return 0


if __name__=="__main__":sys.exit(run())
