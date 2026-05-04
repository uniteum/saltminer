// Browser UI for the saltminer engine. The mining loop, kernel binding, and
// keccak-output unpacking all live in saltminer-engine.js. This file only
// reads/writes form fields, syncs with the URL hash, and surfaces progress.

import {
  parseHex, parseU64, bytesToHex, createPipeline, mine,
  DEFAULT_DISPATCH_SIZE,
} from "./saltminer-engine.mjs";

let device = null;
let pipeline = null;
let hasTimestamps = false;
let running = false;
let stopRequested = false;

const $ = (id) => document.getElementById(id);

function logLine(msg) {
  const el = $("log");
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}

function setStatus(msg) {
  $("status").textContent = msg;
}

async function ensureDevice() {
  if (device) return;
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available. Use a recent Chrome, Edge, or Firefox Nightly with WebGPU enabled.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No GPU adapter available.");
  const requiredFeatures = [];
  if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
  device = await adapter.requestDevice({ requiredFeatures });
  hasTimestamps = device.features.has("timestamp-query");
  device.lost.then((info) => {
    logLine(`device lost: ${info.message}`);
    device = null;
  });
  const info = adapter.info ?? {};
  const desc = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" / ") || "unknown";
  logLine(`GPU: ${desc}${hasTimestamps ? "" : " (no timestamp-query support)"}`);

  const wgsl = await (await fetch("./kernel.wgsl")).text();
  pipeline = await createPipeline(device, wgsl);
}

async function onMine() {
  if (running) return;
  stopRequested = false;
  running = true;
  $("mine").disabled = true;
  $("stop").disabled = false;
  $("result").textContent = "";
  $("log").textContent = "";

  try {
    await ensureDevice();

    const deployer = parseHex($("deployer").value, 20, "deployer");
    const argsHash = parseHex($("argshash").value, 32, "argshash");
    const initcodeHash = parseHex($("initcodehash").value, 32, "initcodehash");
    const mask = parseHex($("mask").value, 20, "mask");
    let target = parseHex($("target").value, 20, "target");
    // Match main.rs: silently drop target bits outside the mask.
    for (let i = 0; i < 20; i++) target[i] &= mask[i];

    const [, , minBig] = parseU64($("min").value);
    const [, , maxBig] = parseU64($("max").value);
    const dispatchSize = parseInt($("dispatch").value, 10) || DEFAULT_DISPATCH_SIZE;

    logLine(`mining range [0x${minBig.toString(16)}, 0x${maxBig.toString(16)}), dispatch_size ${dispatchSize}`);
    setStatus("mining…");

    const started = performance.now();
    const result = await mine({
      device, pipeline, hasTimestamps,
      deployer, argsHash, initcodeHash, mask, target,
      min: minBig, max: maxBig, dispatchSize,
      isStopRequested: () => stopRequested,
      onProgress: ({ tested, cursor, gpuNs }) => {
        const elapsed = (performance.now() - started) / 1000;
        const rate = Number(tested) / Math.max(elapsed, 1e-9);
        const gpuMs = gpuNs ? ` (GPU ${(Number(gpuNs) / 1e6).toFixed(1)} ms/dispatch)` : "";
        setStatus(`tested ${tested.toString()} salts, ${(rate / 1e6).toFixed(2)} MH/s${gpuMs}, next 0x${cursor.toString(16)}`);
      },
    });

    if (result.kind === "found") {
      const addr = bytesToHex(result.address);
      $("result").innerHTML =
        `<div><strong>match</strong></div>` +
        `<div>salt = 0x${result.saltHex}</div>` +
        `<div>home = 0x${addr}</div>`;
      logLine(`match: salt=0x${result.saltHex} home=0x${addr}`);
      setStatus("found");
    } else if (result.kind === "stopped") {
      logLine(`stopped. resume with min=0x${result.cursor.toString(16)}`);
      setStatus("stopped");
    } else {
      logLine("range exhausted without a match");
      setStatus("exhausted");
    }
  } catch (e) {
    logLine(`error: ${e.message}`);
    setStatus("error");
  } finally {
    running = false;
    $("mine").disabled = false;
    $("stop").disabled = true;
  }
}

function onStop() {
  stopRequested = true;
}

const HASH_FIELDS = ["deployer", "initcodehash", "argshash", "mask", "target", "min", "max", "dispatch"];

function applyHashToInputs() {
  const hash = location.hash.startsWith("#") ? location.hash.slice(1) : location.hash;
  if (!hash) return;
  const params = new URLSearchParams(hash);
  for (const id of HASH_FIELDS) {
    if (params.has(id)) $(id).value = params.get(id);
  }
}

function writeInputsToHash() {
  const params = new URLSearchParams();
  for (const id of HASH_FIELDS) params.set(id, $(id).value);
  history.replaceState(null, "", `#${params.toString()}`);
}

window.addEventListener("DOMContentLoaded", () => {
  $("mine").addEventListener("click", onMine);
  $("stop").addEventListener("click", onStop);

  applyHashToInputs();
  writeInputsToHash();
  for (const id of HASH_FIELDS) {
    $(id).addEventListener("input", writeInputsToHash);
  }
  window.addEventListener("hashchange", applyHashToInputs);
});
