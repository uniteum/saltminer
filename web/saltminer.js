// WebGPU host for the saltminer kernel. Mirrors src/main.rs: parse the
// inputs, precompute base_state and mask_lanes the same way src/lib.rs does,
// then dispatch the kernel in chunks and read back the found flag each pass.

const WORKGROUP_SIZE = 256;
const DEFAULT_DISPATCH_SIZE = 1 << 20; // matches the Rust binary's --dispatch default

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

function parseHex(s, byteLen, name) {
  s = s.trim().replace(/^0x/i, "");
  if (s.length !== byteLen * 2) {
    throw new Error(`${name}: expected ${byteLen} bytes (${byteLen * 2} hex chars), got ${s.length}`);
  }
  if (!/^[0-9a-fA-F]+$/.test(s)) {
    throw new Error(`${name}: invalid hex characters`);
  }
  const out = new Uint8Array(byteLen);
  for (let i = 0; i < byteLen; i++) {
    out[i] = parseInt(s.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

// Parse a u64 from decimal or 0x-prefixed hex into a [lo, hi] pair of u32s.
function parseU64(s) {
  s = s.trim();
  let big;
  if (s.startsWith("0x") || s.startsWith("0X")) {
    big = BigInt(s);
  } else {
    big = BigInt(s);
  }
  if (big < 0n || big > 0xffffffffffffffffn) {
    throw new Error(`u64 out of range: ${s}`);
  }
  const lo = Number(big & 0xffffffffn) >>> 0;
  const hi = Number((big >> 32n) & 0xffffffffn) >>> 0;
  return [lo, hi, big];
}

function u64ToHex(lo, hi) {
  const big = (BigInt(hi) << 32n) | BigInt(lo);
  return big.toString(16);
}

// Mirrors lib.rs::compute_base_state. Returns 17 lanes packed as 34 little-
// endian u32s (lo, hi pairs), ready to upload to a storage buffer that the
// shader reads as array<vec2<u32>, 17>.
function computeBaseState(deployer, argsHash, initcodeHash) {
  const msg = new Uint8Array(136);
  msg[0] = 0xff;
  msg.set(deployer, 1);
  msg.set(argsHash, 21);
  msg.set(initcodeHash, 53);
  msg[85] = 0x01;
  msg[135] = 0x80;
  const out = new Uint32Array(34);
  const dv = new DataView(msg.buffer);
  for (let i = 0; i < 17; i++) {
    out[i * 2]     = dv.getUint32(i * 8, true);
    out[i * 2 + 1] = dv.getUint32(i * 8 + 4, true);
  }
  return out;
}

// Mirrors lib.rs::compute_lane_masks. Returns 6 lanes (mask/target × 3) as 12
// u32s, in the order [m1.lo, m1.hi, t1.lo, t1.hi, m2.lo, m2.hi, t2.lo, t2.hi,
// m3.lo, m3.hi, t3.lo, t3.hi]. Lane 1 covers address bytes 0..4 (in the high
// half of state lane 1), lane 2 covers bytes 4..12, lane 3 covers bytes 12..20.
function computeLaneMasks(mask, target) {
  function laneU64FromHighBytes(b) {
    // bytes go into bits 32..63 little-endian: hi = b0 | (b1<<8) | (b2<<16) | (b3<<24)
    const hi = (b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)) >>> 0;
    return [0, hi];
  }
  function laneU64LE(b, off) {
    const lo = (b[off]     | (b[off + 1] << 8) | (b[off + 2] << 16) | (b[off + 3] << 24)) >>> 0;
    const hi = (b[off + 4] | (b[off + 5] << 8) | (b[off + 6] << 16) | (b[off + 7] << 24)) >>> 0;
    return [lo, hi];
  }
  const out = new Uint32Array(12);
  const [m1lo, m1hi] = laneU64FromHighBytes(mask);
  const [t1lo, t1hi] = laneU64FromHighBytes(target);
  out[0] = m1lo; out[1] = m1hi;
  out[2] = t1lo; out[3] = t1hi;
  const [m2lo, m2hi] = laneU64LE(mask, 4);
  const [t2lo, t2hi] = laneU64LE(target, 4);
  out[4] = m2lo; out[5] = m2hi;
  out[6] = t2lo; out[7] = t2hi;
  const [m3lo, m3hi] = laneU64LE(mask, 12);
  const [t3lo, t3hi] = laneU64LE(target, 12);
  out[8]  = m3lo; out[9]  = m3hi;
  out[10] = t3lo; out[11] = t3hi;
  return out;
}

// Mirrors lib.rs::address_from_state. a1/a2/a3 are each [lo, hi] u32 pairs.
function addressFromState(a1, a2, a3) {
  const out = new Uint8Array(20);
  // out[0..4] = bytes 4..7 of lane 1 = bytes 0..3 of a1.hi (little-endian)
  out[0] = a1[1] & 0xff;
  out[1] = (a1[1] >>> 8) & 0xff;
  out[2] = (a1[1] >>> 16) & 0xff;
  out[3] = (a1[1] >>> 24) & 0xff;
  // out[4..12] = a2 little-endian
  out[4]  = a2[0] & 0xff;
  out[5]  = (a2[0] >>> 8) & 0xff;
  out[6]  = (a2[0] >>> 16) & 0xff;
  out[7]  = (a2[0] >>> 24) & 0xff;
  out[8]  = a2[1] & 0xff;
  out[9]  = (a2[1] >>> 8) & 0xff;
  out[10] = (a2[1] >>> 16) & 0xff;
  out[11] = (a2[1] >>> 24) & 0xff;
  // out[12..20] = a3 little-endian
  out[12] = a3[0] & 0xff;
  out[13] = (a3[0] >>> 8) & 0xff;
  out[14] = (a3[0] >>> 16) & 0xff;
  out[15] = (a3[0] >>> 24) & 0xff;
  out[16] = a3[1] & 0xff;
  out[17] = (a3[1] >>> 8) & 0xff;
  out[18] = (a3[1] >>> 16) & 0xff;
  out[19] = (a3[1] >>> 24) & 0xff;
  return out;
}

function bytesToHex(b) {
  return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
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
  const module = device.createShaderModule({ code: wgsl });
  pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

function createSharedBuffers(baseStateU32, maskLanesU32) {
  const baseStateBuffer = device.createBuffer({
    size: baseStateU32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(baseStateBuffer, 0, baseStateU32);
  const maskBuffer = device.createBuffer({
    size: maskLanesU32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(maskBuffer, 0, maskLanesU32);
  return { base: baseStateBuffer, mask: maskBuffer };
}

// Per-slot buffers + bind group. Two slots are kept so we can submit dispatch
// N+1 before awaiting dispatch N's mapAsync, which otherwise stalls the GPU.
function createSlot(shared) {
  const ctrl = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const found = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const result = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const foundRead = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const resultRead = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ctrl } },
      { binding: 1, resource: { buffer: shared.base } },
      { binding: 2, resource: { buffer: shared.mask } },
      { binding: 3, resource: { buffer: found } },
      { binding: 4, resource: { buffer: result } },
    ],
  });
  let querySet = null, queryResolve = null, queryRead = null;
  if (hasTimestamps) {
    querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    queryResolve = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    queryRead = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }
  return { ctrl, found, result, foundRead, resultRead, bindGroup, querySet, queryResolve, queryRead, inflight: false, gpuNs: 0n };
}

function submitDispatch(slot, startLo, startHi, maxLo, maxHi, dispatchSize) {
  device.queue.writeBuffer(slot.ctrl, 0, new Uint32Array([startLo, startHi, maxLo, maxHi]));
  device.queue.writeBuffer(slot.found, 0, new Uint32Array([0]));
  const encoder = device.createCommandEncoder();
  const passDesc = {};
  if (slot.querySet) {
    passDesc.timestampWrites = {
      querySet: slot.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    };
  }
  const pass = encoder.beginComputePass(passDesc);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, slot.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(dispatchSize / WORKGROUP_SIZE));
  pass.end();
  if (slot.querySet) {
    encoder.resolveQuerySet(slot.querySet, 0, 2, slot.queryResolve, 0);
    encoder.copyBufferToBuffer(slot.queryResolve, 0, slot.queryRead, 0, 16);
  }
  encoder.copyBufferToBuffer(slot.found, 0, slot.foundRead, 0, 4);
  encoder.copyBufferToBuffer(slot.result, 0, slot.resultRead, 0, 32);
  device.queue.submit([encoder.finish()]);
  slot.inflight = true;
}

async function awaitSlot(slot) {
  await slot.foundRead.mapAsync(GPUMapMode.READ);
  const found = new Uint32Array(slot.foundRead.getMappedRange().slice(0))[0] !== 0;
  slot.foundRead.unmap();
  slot.inflight = false;
  if (slot.queryRead) {
    await slot.queryRead.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(slot.queryRead.getMappedRange().slice(0));
    slot.gpuNs = ts[1] - ts[0];
    slot.queryRead.unmap();
  }
  if (!found) return null;

  await slot.resultRead.mapAsync(GPUMapMode.READ);
  const r = new Uint32Array(slot.resultRead.getMappedRange().slice(0));
  const out = {
    saltLo: r[0],
    saltHi: r[1],
    a1: [r[2], r[3]],
    a2: [r[4], r[5]],
    a3: [r[6], r[7]],
  };
  slot.resultRead.unmap();
  return out;
}

async function mine() {
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

    const [minLo, minHi, minBig] = parseU64($("min").value);
    const [maxLo, maxHi, maxBig] = parseU64($("max").value);
    if (minBig >= maxBig) throw new Error("min must be < max");

    const dispatchSize = parseInt($("dispatch").value, 10) || DEFAULT_DISPATCH_SIZE;
    if (dispatchSize <= 0) throw new Error("dispatch size must be > 0");

    const baseState = computeBaseState(deployer, argsHash, initcodeHash);
    const maskLanes = computeLaneMasks(mask, target);
    const shared = createSharedBuffers(baseState, maskLanes);
    const slots = [createSlot(shared), createSlot(shared)];

    logLine(`mining range [0x${minBig.toString(16)}, 0x${maxBig.toString(16)}), dispatch_size ${dispatchSize}`);
    setStatus("mining…");

    const started = performance.now();
    let tested = 0n;
    let cursor = minBig;
    const stepBig = BigInt(dispatchSize);

    function tryPrime(slot) {
      if (cursor >= maxBig || stopRequested) return false;
      const startLo = Number(cursor & 0xffffffffn) >>> 0;
      const startHi = Number((cursor >> 32n) & 0xffffffffn) >>> 0;
      submitDispatch(slot, startLo, startHi, maxLo, maxHi, dispatchSize);
      cursor += stepBig;
      return true;
    }

    tryPrime(slots[0]);
    tryPrime(slots[1]);

    let active = 0;
    let result = null;
    while (slots[0].inflight || slots[1].inflight) {
      const slot = slots[active];
      if (!slot.inflight) {
        active = 1 - active;
        continue;
      }

      const r = await awaitSlot(slot);
      if (r) {
        result = r;
        break;
      }

      tested += stepBig;
      const elapsed = (performance.now() - started) / 1000;
      const rate = Number(tested) / Math.max(elapsed, 1e-9);
      const gpuMs = slot.queryRead ? ` (GPU ${(Number(slot.gpuNs) / 1e6).toFixed(1)} ms/dispatch)` : "";
      setStatus(`tested ${tested.toString()} salts, ${(rate / 1e6).toFixed(2)} MH/s${gpuMs}, next 0x${cursor.toString(16)}`);

      tryPrime(slot);
      active = 1 - active;
    }

    if (result) {
      const saltHex = u64ToHex(result.saltLo, result.saltHi).padStart(64, "0");
      const addr = bytesToHex(addressFromState(result.a1, result.a2, result.a3));
      $("result").innerHTML =
        `<div><strong>match</strong></div>` +
        `<div>salt = 0x${saltHex}</div>` +
        `<div>home = 0x${addr}</div>`;
      logLine(`match: salt=0x${saltHex} home=0x${addr}`);
      setStatus("found");
    } else if (stopRequested) {
      // The drain loop above awaits every still-inflight slot before exiting,
      // so everything submitted has already been verified. Cursor sits at the
      // first salt that was never dispatched.
      logLine(`stopped. resume with min=0x${cursor.toString(16)}`);
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

function stop() {
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
  // Use replaceState so editing fields doesn't spam history.
  history.replaceState(null, "", `#${params.toString()}`);
}

window.addEventListener("DOMContentLoaded", () => {
  $("mine").addEventListener("click", mine);
  $("stop").addEventListener("click", stop);

  applyHashToInputs();
  writeInputsToHash();
  for (const id of HASH_FIELDS) {
    $(id).addEventListener("input", writeInputsToHash);
  }
  window.addEventListener("hashchange", applyHashToInputs);
});
