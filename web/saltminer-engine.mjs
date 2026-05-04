// Pure WebGPU mining engine. No DOM, no Node APIs. Both the browser shim
// (saltminer.js) and the Node CLI import this module.
//
// The engine takes an already-acquired GPUDevice plus the kernel.wgsl source
// as a string. The caller is responsible for fetching/reading the WGSL —
// fetch() in the browser, fs.readFile in Node — so this module stays platform
// agnostic.

const WORKGROUP_SIZE = 256;
export const DEFAULT_DISPATCH_SIZE = 1 << 20;

export function parseHex(s, byteLen, name) {
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

// Parse a u64 from decimal or 0x-prefixed hex into [lo, hi, big] (u32, u32, BigInt).
export function parseU64(s) {
  s = s.trim();
  const big = BigInt(s);
  if (big < 0n || big > 0xffffffffffffffffn) {
    throw new Error(`u64 out of range: ${s}`);
  }
  const lo = Number(big & 0xffffffffn) >>> 0;
  const hi = Number((big >> 32n) & 0xffffffffn) >>> 0;
  return [lo, hi, big];
}

export function u64ToHex(lo, hi) {
  return ((BigInt(hi) << 32n) | BigInt(lo)).toString(16);
}

export function bytesToHex(b) {
  return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
}

// Mirrors lib.rs::compute_base_state. Returns 17 lanes packed as 34 little-
// endian u32s (lo, hi pairs), ready to upload to a storage buffer the shader
// reads as array<vec2<u32>, 17>.
export function computeBaseState(deployer, argsHash, initcodeHash) {
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

// Mirrors lib.rs::compute_lane_masks. Returns 6 lanes (mask/target × 3) as 12 u32s.
export function computeLaneMasks(mask, target) {
  function laneU64FromHighBytes(b) {
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
export function addressFromState(a1, a2, a3) {
  const out = new Uint8Array(20);
  out[0] = a1[1] & 0xff;
  out[1] = (a1[1] >>> 8) & 0xff;
  out[2] = (a1[1] >>> 16) & 0xff;
  out[3] = (a1[1] >>> 24) & 0xff;
  out[4]  = a2[0] & 0xff;
  out[5]  = (a2[0] >>> 8) & 0xff;
  out[6]  = (a2[0] >>> 16) & 0xff;
  out[7]  = (a2[0] >>> 24) & 0xff;
  out[8]  = a2[1] & 0xff;
  out[9]  = (a2[1] >>> 8) & 0xff;
  out[10] = (a2[1] >>> 16) & 0xff;
  out[11] = (a2[1] >>> 24) & 0xff;
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

export async function createPipeline(device, wgsl) {
  const module = device.createShaderModule({ code: wgsl });
  return await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

function createSharedBuffers(device, baseStateU32, maskLanesU32) {
  const base = device.createBuffer({
    size: baseStateU32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(base, 0, baseStateU32);
  const mask = device.createBuffer({
    size: maskLanesU32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(mask, 0, maskLanesU32);
  return { base, mask };
}

// Per-slot buffers + bind group. Two slots are kept so dispatch N+1 can be
// submitted before awaiting dispatch N's mapAsync, which otherwise stalls the GPU.
function createSlot(device, pipeline, shared, hasTimestamps) {
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

function submitDispatch(device, pipeline, slot, startLo, startHi, maxLo, maxHi, dispatchSize) {
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

// Run the mining loop. Returns one of:
//   { kind: "found", saltLo, saltHi, a1, a2, a3, address, saltHex }
//   { kind: "stopped", cursor: BigInt }
//   { kind: "exhausted", cursor: BigInt }
//
// `cfg`:
//   device, pipeline, hasTimestamps   — already created by the caller
//   deployer, argsHash, initcodeHash, mask, target  — Uint8Arrays (20/32/32/20/20)
//   min, max                          — BigInt, half-open
//   dispatchSize                      — Number, threads per dispatch
//   isStopRequested()                 — optional, called between dispatches
//   onProgress({ tested, cursor, gpuNs }) — optional, called after each dispatch
export async function mine(cfg) {
  const {
    device, pipeline, hasTimestamps,
    deployer, argsHash, initcodeHash, mask, target,
    min, max, dispatchSize,
    isStopRequested = () => false,
    onProgress = () => {},
  } = cfg;

  if (min >= max) throw new Error("min must be < max");
  if (dispatchSize <= 0) throw new Error("dispatch size must be > 0");

  const baseState = computeBaseState(deployer, argsHash, initcodeHash);
  const maskLanes = computeLaneMasks(mask, target);
  const shared = createSharedBuffers(device, baseState, maskLanes);
  const slots = [
    createSlot(device, pipeline, shared, hasTimestamps),
    createSlot(device, pipeline, shared, hasTimestamps),
  ];

  const maxLo = Number(max & 0xffffffffn) >>> 0;
  const maxHi = Number((max >> 32n) & 0xffffffffn) >>> 0;
  const stepBig = BigInt(dispatchSize);
  let cursor = min;
  let tested = 0n;

  function tryPrime(slot) {
    if (cursor >= max || isStopRequested()) return false;
    const startLo = Number(cursor & 0xffffffffn) >>> 0;
    const startHi = Number((cursor >> 32n) & 0xffffffffn) >>> 0;
    submitDispatch(device, pipeline, slot, startLo, startHi, maxLo, maxHi, dispatchSize);
    cursor += stepBig;
    return true;
  }

  tryPrime(slots[0]);
  tryPrime(slots[1]);

  let active = 0;
  while (slots[0].inflight || slots[1].inflight) {
    const slot = slots[active];
    if (!slot.inflight) {
      active = 1 - active;
      continue;
    }
    const r = await awaitSlot(slot);
    if (r) {
      const address = addressFromState(r.a1, r.a2, r.a3);
      const saltHex = u64ToHex(r.saltLo, r.saltHi).padStart(64, "0");
      return { kind: "found", ...r, address, saltHex };
    }
    tested += stepBig;
    onProgress({ tested, cursor, gpuNs: slot.gpuNs });
    tryPrime(slot);
    active = 1 - active;
  }

  return {
    kind: isStopRequested() ? "stopped" : "exhausted",
    cursor,
  };
}
