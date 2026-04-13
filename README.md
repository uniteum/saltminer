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

The search is bounded: `salt ∈ [min, max)`. A caller who has already exhausted one range and wants to continue just bumps `min` upward and relaunches.

**v1 simplification: salt is a `u64`, not a `uint256`.** Only the low 8 bytes of the on-chain `create2Salt` actually vary across attempts — those 8 bytes occupy the position Solidity would write `bytes32(uint256(salt))` to, i.e. the last 8 bytes of the 32-byte salt word. This keeps the GPU kernel working in native 64-bit arithmetic with no big-integer code, and 2⁶⁴ ≈ 1.8 × 10¹⁹ attempts is vastly more than any realistic mining budget. The other 24 bytes of `create2Salt` are fixed to the corresponding bytes of `argsHash`, so the address still binds to every parameter committed in `argsHash`. Widening the salt beyond 64 bits is a future extension and would only touch the kernel's salt-unpacking lines and the host's range arithmetic.

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

A single `saltminer` worker saturates one whole GPU on its own. This section explains how, because the GPU execution model is unlike a CPU for-loop and the difference matters for understanding the sharding math below.

### The kernel is not a loop

On a CPU you'd test a million salts like this:

```c
for (size_t i = 0; i < 1048576; i++) {
    uint64_t salt = start_salt + i * stride;
    if (matches(salt)) report(salt);
}
```

On the GPU there is no `for`. The kernel is written as the *body* of that loop only — one iteration, parameterized by a thread ID:

```c
__kernel void mine(ulong start_salt, ulong stride, /* ... */) {
    size_t i    = get_global_id(0);       // "which iteration am I?"
    ulong  salt = start_salt + i * stride;
    if (matches(salt)) report(salt);
}
```

The host then tells the driver **"run this kernel 1,048,576 times in parallel"** by passing `global_size = 1 << 20` to the enqueue call. OpenCL spawns 1,048,576 logical threads, each of which sees a different `i` from `get_global_id(0)`. There is no explicit iteration anywhere: the loop has been *unrolled into hardware parallelism*. Every thread runs the body once, in parallel, on its own salt.

So when this README says "thread `i` computes `salt = start_salt + i * stride`," it means: the kernel body runs `global_size` times simultaneously, and the `i`-th invocation picks up the `i`-th salt.

### How those threads land on physical cores

A GPU does not have a million physical cores. A modest integrated iGPU might have ~100 execution units; a big discrete card has a few thousand shader cores. The driver maps the 1M logical threads onto that physical hardware in three steps, none of which the host code has to think about:

1. **Group threads into warps / wavefronts.** The 1M threads are partitioned into fixed-size bundles (32 threads on NVIDIA, 32 or 64 on AMD, 8–32 on Intel). Every thread in a bundle executes the same instruction at the same time on different data — this is SIMD inside the GPU.
2. **Schedule bundles onto compute units.** The driver hands each compute unit a queue of bundles. A compute unit runs one bundle's instruction, then swaps to another bundle while the first waits on memory or a long-latency op. This overlap is how GPUs hide latency and keep all cores busy.
3. **Iterate until every bundle is done.** A compute unit chews through its assigned bundles one after another. When all 1M threads have finished, the dispatch is complete and control returns to the host.

The host code never participates in any of this. It writes a kernel that processes one salt, picks a `global_size`, and enqueues it. The driver handles bundling, scheduling, latency hiding, and mapping onto however many cores the device actually has. The same binary saturates a 96-EU Intel iGPU and a 16,000-core discrete GPU without change.

### Where the outer loop comes back

A single dispatch only covers `global_size` salts — a million or so. A search range can be much larger than that, so the host wraps the dispatch in an outer loop:

```
start_salt = min + w
while start_salt < max:
    enqueue_kernel(start_salt, stride, ...)   // runs global_size threads in parallel on the GPU
    wait_for_completion()
    start_salt += global_size * stride        // advance to the next million salts
```

Each iteration of this *host* loop launches one GPU dispatch, which internally runs a million parallel kernel invocations, which together cover one million salts. The host loop exists so Ctrl-C can be noticed between dispatches and so `start_salt` can be advanced; it is not the parallelism, it is the chunking.

Three levels total, from innermost to outermost:

1. **Inside one dispatch:** `global_size` parallel kernel invocations on the GPU, each testing one salt. The driver places them on physical cores.
2. **Across dispatches in one worker:** the host loop issues dispatch after dispatch, advancing `start_salt` by `global_size * stride` each time, until the range is exhausted or a hit is found.
3. **Across workers in one job:** the `--shard w/N` scheme (next section) splits a search range across `N` independent worker processes.

The GPU's multi-core parallelism lives entirely at level 1. There is no sharding to do across GPU cores: pick a `global_size` large enough to fill the device and let the driver place the threads.

### Tuning `global_size` and `local_size`

- **`global_size`** — how many parallel kernel invocations per dispatch. Want this large enough that every compute unit on the GPU has bundles to chew on (so none sit idle), but not so large that one dispatch takes long enough to make Ctrl-C feel laggy. A good starting point is `2^20` (about a million threads per dispatch); tune from there.
- **`local_size`** — how many of those threads are bundled into one work-group. OpenCL reports a device-preferred multiple via `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`; using that (or a small multiple like 64 or 256) is almost always right. The host queries the driver rather than hardcoding.

These knobs affect throughput only. The salt assignment math (`salt = start_salt + i * stride`) is independent of both.

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
- `--min`, `--max` — half-open `u64` salt search range. See the "Salt space" section above for why v1 uses `u64`.
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
