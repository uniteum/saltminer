# saltminer (web)

WebGPU port of the OpenCL kernel in `src/kernel.cl`. Same math, same byte
layout, same output format — just dispatched from JS instead of Rust.

## Run

WebGPU isn't allowed via `file://`, so serve the directory over HTTP:

    python3 -m http.server -d web 8000

Then open <http://localhost:8000/>. Needs Chrome/Edge 113+ or Firefox with
WebGPU enabled.

## Notes

- Single-browser only — no `--shard` flag. Run separate browser tabs with
  disjoint `min`/`max` ranges if you want to split work.
- `kernel.wgsl` and `src/kernel.cl` should stay in sync. The salt-injection
  math (the lane 5/6 XOR) and the mask comparison must match exactly or the
  two miners will diverge.
