#!/usr/bin/env node
// Node CLI for the saltminer engine. Reuses web/saltminer-engine.js verbatim,
// substituting fs.readFile for fetch() and @kmamal/gpu (Dawn) for navigator.gpu.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import gpu from "@kmamal/gpu";
import {
  parseHex, parseU64, bytesToHex, createPipeline, mine,
  DEFAULT_DISPATCH_SIZE,
} from "../web/saltminer-engine.mjs";

// The engine references GPUBufferUsage and GPUMapMode as globals (same as the
// browser). @kmamal/gpu exposes them as module fields; install on globalThis.
globalThis.GPUBufferUsage = gpu.GPUBufferUsage;
globalThis.GPUMapMode = gpu.GPUMapMode;
globalThis.GPUShaderStage = gpu.GPUShaderStage;

function usage() {
  process.stderr.write(`saltminer (Node) — GPU-accelerated CREATE2 salt miner.

Usage:
  saltminer --deployer <hex> --initcodehash <hex> --argshash <hex> \\
            --mask <hex> --target <hex> [--min <u64>] [--max <u64>] \\
            [--dispatch <n>] [--listdevices]

  --deployer       Factory address, 20-byte hex.
  --initcodehash   keccak256 of init code, 32-byte hex.
  --argshash       keccak256(abi.encode(...)) of bound args, 32-byte hex.
  --mask           20-byte address bit-mask.
  --target         20-byte address target value.
  --min            Inclusive lower bound for the u64 salt search (default 0).
  --max            Exclusive upper bound (default 0xffffffffffffffff).
  --dispatch       Threads per kernel dispatch (default ${DEFAULT_DISPATCH_SIZE}).
  --listdevices    List available WebGPU adapters and exit.

Notes:
  - This Node version uses the same WGSL kernel as web/. Shard (--shard w/N)
    is not implemented here; the Rust binary remains the multi-process option.
`);
}

function parseArgs(argv) {
  const out = {
    min: "0",
    max: "0xffffffffffffffff",
    dispatch: String(DEFAULT_DISPATCH_SIZE),
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") { out.help = true; continue; }
    if (a === "--listdevices") { out.listDevices = true; continue; }
    if (!a.startsWith("--")) throw new Error(`unexpected arg: ${a}`);
    const key = a.slice(2);
    const val = argv[++i];
    if (val === undefined) throw new Error(`${a} expects a value`);
    out[key] = val;
  }
  return out;
}

async function listDevices(navigatorGpu) {
  const adapter = await navigatorGpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    console.log("no GPU adapter available");
    return;
  }
  const info = adapter.info ?? {};
  const desc = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean).join(" / ") || "unknown";
  console.log(`high-performance: ${desc}`);
  const lp = await navigatorGpu.requestAdapter({ powerPreference: "low-power" });
  if (lp && lp !== adapter) {
    const li = lp.info ?? {};
    const ldesc = [li.vendor, li.architecture, li.device, li.description]
      .filter(Boolean).join(" / ") || "unknown";
    console.log(`low-power:        ${ldesc}`);
  }
}

async function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (e) {
    process.stderr.write(`error: ${e.message}\n\n`);
    usage();
    process.exit(2);
  }
  if (args.help) { usage(); process.exit(0); }

  const navigatorGpu = gpu.create([]);

  if (args.listDevices) {
    await listDevices(navigatorGpu);
    process.exit(0);
  }

  for (const k of ["deployer", "initcodehash", "argshash", "mask", "target"]) {
    if (!args[k]) {
      process.stderr.write(`error: --${k} required\n\n`);
      usage();
      process.exit(2);
    }
  }

  const deployer = parseHex(args.deployer, 20, "deployer");
  const argsHash = parseHex(args.argshash, 32, "argshash");
  const initcodeHash = parseHex(args.initcodehash, 32, "initcodehash");
  const mask = parseHex(args.mask, 20, "mask");
  const target = parseHex(args.target, 20, "target");
  for (let i = 0; i < 20; i++) target[i] &= mask[i];

  const [, , minBig] = parseU64(args.min);
  const [, , maxBig] = parseU64(args.max);
  if (minBig >= maxBig) {
    process.stderr.write("error: --min must be < --max\n");
    process.exit(2);
  }
  const dispatchSize = parseInt(args.dispatch, 10);
  if (!(dispatchSize > 0)) {
    process.stderr.write("error: --dispatch must be > 0\n");
    process.exit(2);
  }

  const adapter = await navigatorGpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no GPU adapter available");
  const requiredFeatures = [];
  if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({ requiredFeatures });
  const hasTimestamps = device.features.has("timestamp-query");
  const info = adapter.info ?? {};
  const desc = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean).join(" / ") || "unknown";
  process.stderr.write(`saltminer: GPU ${desc}${hasTimestamps ? "" : " (no timestamp-query)"}\n`);
  process.stderr.write(`saltminer: range [0x${minBig.toString(16)}, 0x${maxBig.toString(16)}), dispatch ${dispatchSize}\n`);

  const here = dirname(fileURLToPath(import.meta.url));
  const wgsl = await readFile(join(here, "..", "web", "kernel.wgsl"), "utf8");
  const pipeline = await createPipeline(device, wgsl);

  let stopRequested = false;
  process.on("SIGINT", () => { stopRequested = true; });

  const started = process.hrtime.bigint();
  let lastReport = 0n;
  const result = await mine({
    device, pipeline, hasTimestamps,
    deployer, argsHash, initcodeHash, mask, target,
    min: minBig, max: maxBig, dispatchSize,
    isStopRequested: () => stopRequested,
    onProgress: ({ tested, cursor, gpuNs }) => {
      const now = process.hrtime.bigint();
      // Throttle stderr writes — mining at 100M+ H/s, one line per dispatch is fine,
      // but this also avoids flooding when terminal scrolling is the bottleneck.
      if (now - lastReport < 100_000_000n) return;
      lastReport = now;
      const elapsedSec = Number(now - started) / 1e9;
      const rate = Number(tested) / Math.max(elapsedSec, 1e-9);
      const gpuMs = gpuNs ? ` (GPU ${(Number(gpuNs) / 1e6).toFixed(1)} ms/dispatch)` : "";
      process.stderr.write(`\rsaltminer: tested ${tested} salts, ${(rate / 1e6).toFixed(2)} MH/s${gpuMs}, next 0x${cursor.toString(16)}  `);
    },
  });
  process.stderr.write("\n");

  if (result.kind === "found") {
    const addr = bytesToHex(result.address);
    console.log(`salt = 0x${result.saltHex}`);
    console.log(`home = 0x${addr}`);
    process.exit(0);
  } else if (result.kind === "stopped") {
    process.stderr.write(`saltminer: interrupted. resume with --min 0x${result.cursor.toString(16)} --max 0x${maxBig.toString(16)}\n`);
    process.exit(1);
  } else {
    process.stderr.write(`saltminer: range exhausted without a match. widen with --min 0x${result.cursor.toString(16)} --max <new>\n`);
    process.exit(1);
  }
}

main().catch((e) => {
  process.stderr.write(`error: ${e.stack || e.message}\n`);
  process.exit(1);
});
