# saltminer (node)

Node CLI on top of the same WGSL kernel as `web/`. Reuses
`web/saltminer-engine.mjs` for the dispatch loop and lane math, swaps
`fetch` for `fs` and `navigator.gpu` for `@kmamal/gpu` (a Node-Dawn binding).

## Run

    cd node
    npm install            # pulls a prebuilt Dawn binary (~30 MB) into node_modules
    node saltminer.js --help

## WSL2 caveat

`@kmamal/gpu`'s standalone Dawn build needs a Vulkan ICD that talks to
the iGPU. WSL2 ships `/dev/dxg` (D3D12 passthrough) but no `/dev/dri`,
so the Intel and AMD Vulkan ICDs from `mesa-vulkan-drivers` find nothing
and Dawn falls back to `llvmpipe` (Mesa's CPU rasterizer). Throughput
collapses to single-digit MH/s.

The fix is Mesa's `dzn` driver (Vulkan-on-D3D12), which Ubuntu 24.04
does not package. On WSL2, prefer the Rust binary — its OpenCL stack
reaches the iGPU via the Intel OpenCL runtime, which works without
`/dev/dri`.

Browsers sidestep this because Chrome/Edge ship their own GPU stack
that talks to `/dev/dxg` directly.

## Notes

- No `--shard` flag (kernel uses `salt = start_salt + i` with no stride).
  The Rust binary remains the multi-process option.
- `kernel.wgsl` lives in `web/`; this CLI reads it from there at startup.
