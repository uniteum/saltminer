# saltminer

A GPU-accelerated CREATE2 salt miner. Given a deployer address, an init-code hash, and an arguments hash, `saltminer` searches for a 256-bit salt such that the resulting CREATE2 address matches a caller-supplied bit pattern.

Inspired by [0age/create2crunch](https://github.com/0age/create2crunch), generalized to arbitrary bit-mask matching and to contracts that bind deployment parameters into the salt via a precomputed arguments hash.

## Problem

A factory contract deploys child contracts with CREATE2. The child address depends on the deployer, the init code, and a 32-byte salt. A caller who wants a vanity address — leading zeros, a recognizable prefix, embedded nibbles, whatever — needs to search the salt space until the predicted address matches their desired pattern.

The search is embarrassingly parallel and dominated by keccak-256. A GPU is the right tool.

## Matching criteria

`saltminer` treats the 160-bit address as an integer and accepts a `(mask, match)` pair:

```
(uint160(home) & mask) == match
```

This subsumes every common vanity pattern:

| Pattern                | Mask                                         | Match                                        |
| ---------------------- | -------------------------------------------- | -------------------------------------------- |
| 4 leading zero bytes   | `0xffffffff00000000000000000000000000000000` | `0x0000000000000000000000000000000000000000` |
| Trailing `...dead`     | `0x000000000000000000000000000000000000ffff` | `0x000000000000000000000000000000000000dead` |
| Specific middle nibble | (bit at position)                            | (bit at position)                            |

Any bit the caller doesn't care about is simply zero in the mask.

## Salt space

The search is bounded: `salt ∈ [min, max)`, where `min` and `max` are `uint256` values. A caller who wants to search the full range just passes `0` and `2^256`. A caller who has already exhausted one range and wants to continue just bumps `min` upward and relaunches.

## Binding deployment parameters to the salt

`saltminer` is aimed at factory contracts whose salt commits to deployment-time parameters. The supported shape is:

```solidity
bytes32 argsHash    = keccak256(abi.encode(/* deployment parameters */));
bytes32 create2Salt = argsHash ^ salt;
```

That is: the contract precomputes `argsHash` over whatever parameters it wants to bind into the address (maker, name, decimals, owner, whatever), then XORs the user-supplied `salt`. The resulting address still binds to every parameter via `argsHash`, the full 256-bit salt space is preserved for mining, and the kernel does exactly one keccak per attempt — the CREATE2 address hash — with no pre-hash on top.

A factory that instead hashes `salt` together with its parameters — `create2Salt = keccak256(abi.encode(params, salt))` — doubles the per-attempt hashing cost on the GPU and is harder to optimize. If you control the contract, prefer the XOR form.

`saltminer` itself is agnostic to how `argsHash` is computed. The caller precomputes it off-chain and passes it in as a 32-byte hex value.

## Algorithm

For a given deployer, init-code hash, args hash, mask, match, and salt range, each GPU thread runs:

```
create2Salt = argsHash ^ salt
home = keccak256(0xff ‖ deployer ‖ create2Salt ‖ initcodeHash)[12:]
if (uint160(home) & mask) == match:
    report (salt, home)
```

**First hit wins.** On a match, the kernel atomically writes the result to a host-visible buffer, the host prints `salt` and `home`, and the program exits.

The init code hash is computed once by the caller — typically `keccak256(deployment_bytecode)`, or for EIP-1167 minimal proxies `keccak256(0x3d602d80600a3d3981f3363d3d373d3d3d363d73 ‖ implementation ‖ 5af43d82803e903d91602b57fd5bf3)`.

## Intra-worker parallelism

A single `saltminer` worker already saturates one whole GPU. Understanding how requires separating three levels of parallelism, from innermost to outermost:

1. **GPU threads within one kernel dispatch.** Each dispatch launches `global_size` threads — typically `2^20` or more. OpenCL distributes them automatically across every compute unit on the device (every SM on NVIDIA, every CU on AMD, every execution unit on an Intel iGPU). The host never schedules onto cores directly; the driver does. This is what makes a single worker scale from a 96-EU integrated iGPU to a discrete GPU with hundreds of CUs without any code change.

2. **Dispatches in sequence.** The worker's host loop issues one dispatch after another, each advancing `start_salt` by `global_size * stride`. Between dispatches the host checks for hits and Ctrl-C. Dispatches are the unit at which the host re-enters control; within a dispatch, the GPU runs uninterrupted.

3. **Workers across the whole job.** The `--shard w/N` scheme splits a search range across `N` independent worker processes, as described below. One worker per GPU (or per machine) is the usual mapping.

So: **the GPU's multi-core parallelism is handled transparently inside level 1.** You do not shard across GPU cores; you pick a `global_size` large enough to keep every core busy and let OpenCL place the threads.

### Tuning `global_size` and `local_size`

- **`global_size`** — total threads per dispatch. Want this large enough that the GPU's schedulers are never idle, but not so large that a single dispatch takes long enough to make Ctrl-C feel laggy. A good starting point is `global_size = 2^20` (about a million threads per dispatch); tune from there.
- **`local_size`** — threads per work-group. OpenCL reports a device-preferred multiple via `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`; picking that (or a small multiple, e.g. 64 or 256) is almost always right. The host asks the driver rather than hardcoding.

These are knobs on the worker, not on the sharding scheme. The salt assignment math (`salt = start_salt + i * stride`) is the same regardless of `global_size`.

### Device selection and multi-GPU hosts

`saltminer` targets one OpenCL device per worker process, selected with a `--device` flag (platform index + device index, listed via `--list-devices`). A host with two GPUs runs two worker processes, each with its own `--device` and its own `--shard w/N`. Simple, crash-isolates one GPU from the other, and uses exactly the same sharding mechanism as multi-machine deployments.

A single-process multi-GPU mode (one command queue per device inside one binary) is **out of scope for v1** — it adds context-management and signal-handling complexity without changing the search itself. If it turns out to be useful later, it can be layered on without touching the kernel or the sharding math.

## Sharding

`saltminer` supports interleaved sharding so a fixed search range can be split across `N` cooperating workers (multiple machines, multiple GPUs on one machine, or multiple processes on one GPU).

Worker `w` of `N` is responsible for the salts

```
{ min + w, min + w + N, min + w + 2N, ... } ∩ [min, max)
```

That is: each worker claims one residue class modulo `N`, starting at offset `w`. Given three workers over `[min, max)`:

- worker 0 tests `min, min+3, min+6, ...`
- worker 1 tests `min+1, min+4, min+7, ...`
- worker 2 tests `min+2, min+5, min+8, ...`

### Why interleaved rather than contiguous

The search is uniform — every salt is equally likely to produce a hit. Both interleaved and contiguous sharding are equivalent in expected time to first hit. Interleaved is preferred here because:

- **Resuming is per-worker.** Each worker tracks only its own next-salt pointer. Extending the upper bound from `max` to `max + K` is a relaunch of each worker with the new `max`; no range re-partitioning.
- **Uniform coverage if a worker dies.** Killing one of three workers still leaves the surviving two evenly spread across the range, not stranded on two thirds of it.

### Tradeoff

Interleaved sharding locks in the worker count `N` at launch. Mid-search you cannot add a fourth worker without disturbing the residue-class assignment of the first three. If you need elastic worker counts, assign each new worker a fresh disjoint sub-range of `[min, max)` instead.

### Kernel parameterization

Host loop for worker `w` of `N`:

```
start_salt = min + w               // first salt this worker ever tests
stride     = N
while start_salt < max:
    dispatch_kernel(start_salt, stride, max, argsHash, deployer, initcodeHash, mask, match)
    start_salt += global_size * stride
```

Thread `i` in a dispatch computes:

```
salt = start_salt + i * stride
if salt >= max: return
// ... test salt ...
```

`start_salt` is "the first salt the next dispatch will test." It starts at `min + w` and marches upward by `global_size * N` per dispatch. `min` is only used to seed the initial value.

### Resume on exit

On `Ctrl-C` or range exhaustion without a hit, the host prints the current `start_salt` for its worker. Restarting the same worker with `--min (start_salt - w)` and the same `--shard w/N` resumes exactly where the previous run left off.

## CLI

```
saltminer \
  --deployer      0x<20-byte-hex> \
  --initcode-hash 0x<32-byte-hex> \
  --args-hash     0x<32-byte-hex> \
  --mask          0xffffffff00000000000000000000000000000000 \
  --match         0x0000000000000000000000000000000000000000 \
  --min           0 \
  --max           0xffffffffffffffff \
  --shard         0/1
```

- `--deployer` — the factory address that will call CREATE2.
- `--initcode-hash` — `keccak256` of the init code the factory will deploy. Precomputed by the caller.
- `--args-hash` — `keccak256(abi.encode(...))` over whatever parameters the factory binds into its salt. Precomputed off-chain; keeps the miner agnostic to parameter layout.
- `--mask`, `--match` — 160-bit vanity criterion.
- `--min`, `--max` — half-open salt search range.
- `--shard` — `w/N`, interleaved worker index and count. Default `0/1` (no sharding).

On a hit, the program prints:

```
salt = 0x...
home = 0x...
```

and exits with status 0. On range exhaustion without a hit, it prints the current `start_salt` and exits non-zero so the caller can resume or widen the range.

## Stack

- **Rust** host driver.
- **OpenCL** kernel via the `ocl` crate. OpenCL is portable across integrated GPUs, discrete AMD/NVIDIA GPUs, and CPUs, which matches the requirement that the tool run anywhere including integrated graphics.
- Kernel keccak-256 borrowed from `create2crunch`, with the init-code hash precomputed on the host and burned into kernel args.

## Non-goals

- **Streaming multiple hits.** First match, exit.
- **Coordination between workers.** Workers are independent; sharding is by pre-agreed `w/N`. No network, no shared state.
- **Automatic load balancing.** Each worker runs its residue class to completion.
- **Signing or broadcasting the deployment transaction.** `saltminer` only produces the salt; the caller submits the transaction.
