// WebGPU port of src/kernel.cl. Each invocation tries one salt:
// reconstruct the 136-byte CREATE2 keccak input from the host-precomputed
// 17-lane base_state, XOR the salt into lanes 5/6 the same way the OpenCL
// kernel does, run keccak-f[1600], and check the address lanes against the
// caller's mask/target. WGSL has no u64, so every lane is a vec2<u32> with
// .x = low 32 bits and .y = high 32 bits.
//
// The keccak inner loops are unrolled with literal PILN/ROTC values so the
// compiler can specialize each rotl64 to a single shift pair instead of a
// variable-rotation routine with branches. The outer 24-round loop is kept
// — unrolling that too would balloon the shader without much extra win.

const RC: array<vec2<u32>, 24> = array<vec2<u32>, 24>(
    vec2<u32>(0x00000001u, 0x00000000u),
    vec2<u32>(0x00008082u, 0x00000000u),
    vec2<u32>(0x0000808au, 0x80000000u),
    vec2<u32>(0x80008000u, 0x80000000u),
    vec2<u32>(0x0000808bu, 0x00000000u),
    vec2<u32>(0x80000001u, 0x00000000u),
    vec2<u32>(0x80008081u, 0x80000000u),
    vec2<u32>(0x00008009u, 0x80000000u),
    vec2<u32>(0x0000008au, 0x00000000u),
    vec2<u32>(0x00000088u, 0x00000000u),
    vec2<u32>(0x80008009u, 0x00000000u),
    vec2<u32>(0x8000000au, 0x00000000u),
    vec2<u32>(0x8000808bu, 0x00000000u),
    vec2<u32>(0x0000008bu, 0x80000000u),
    vec2<u32>(0x00008089u, 0x80000000u),
    vec2<u32>(0x00008003u, 0x80000000u),
    vec2<u32>(0x00008002u, 0x80000000u),
    vec2<u32>(0x00000080u, 0x80000000u),
    vec2<u32>(0x0000800au, 0x00000000u),
    vec2<u32>(0x8000000au, 0x80000000u),
    vec2<u32>(0x80008081u, 0x80000000u),
    vec2<u32>(0x00008080u, 0x80000000u),
    vec2<u32>(0x80000001u, 0x00000000u),
    vec2<u32>(0x80008008u, 0x80000000u),
);

// ctrl = (start_salt.lo, start_salt.hi, max_salt.lo, max_salt.hi)
@group(0) @binding(0) var<uniform> ctrl: vec4<u32>;
@group(0) @binding(1) var<storage, read> base_state: array<vec2<u32>, 17>;
@group(0) @binding(2) var<storage, read> mask_lanes: array<vec2<u32>, 6>;
@group(0) @binding(3) var<storage, read_write> found: atomic<u32>;
// result layout: [salt_lo, salt_hi, a1_lo, a1_hi, a2_lo, a2_hi, a3_lo, a3_hi]
@group(0) @binding(4) var<storage, read_write> result: array<u32, 8>;

fn rotl64(x: vec2<u32>, n: u32) -> vec2<u32> {
    var lo = x.x;
    var hi = x.y;
    if ((n & 32u) != 0u) {
        let t = lo; lo = hi; hi = t;
    }
    let m = n & 31u;
    if (m == 0u) {
        return vec2<u32>(lo, hi);
    }
    let r = 32u - m;
    return vec2<u32>((lo << m) | (hi >> r), (hi << m) | (lo >> r));
}

fn bswap32(x: u32) -> u32 {
    return ((x & 0x000000FFu) << 24u) |
           ((x & 0x0000FF00u) << 8u)  |
           ((x & 0x00FF0000u) >> 8u)  |
           ((x & 0xFF000000u) >> 24u);
}

fn less_u64(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y < b.y) { return true; }
    if (a.y > b.y) { return false; }
    return a.x < b.x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // salt = start_salt + gid.x (u64 + u32, single carry)
    let i = gid.x;
    let salt_lo = ctrl.x + i;
    let carry: u32 = select(0u, 1u, salt_lo < ctrl.x);
    let salt_hi = ctrl.y + carry;
    let salt = vec2<u32>(salt_lo, salt_hi);

    if (!less_u64(salt, vec2<u32>(ctrl.z, ctrl.w))) {
        return;
    }

    var st: array<vec2<u32>, 25>;
    for (var k: u32 = 0u; k < 17u; k = k + 1u) {
        st[k] = base_state[k];
    }
    for (var k: u32 = 17u; k < 25u; k = k + 1u) {
        st[k] = vec2<u32>(0u, 0u);
    }

    // Salt injection, byte-for-byte equivalent to src/kernel.cl:
    //   bs = bswap64(salt)
    //   st[5] ^= (bs & 0xFFFFFF) << 40
    //   st[6] ^= bs >> 24
    // bswap64 on (lo, hi) yields (bswap32(hi), bswap32(lo)).
    let bs_lo = bswap32(salt_hi);
    let bs_hi = bswap32(salt_lo);
    st[5].y = st[5].y ^ ((bs_lo & 0x00FFFFFFu) << 8u);
    st[6].x = st[6].x ^ ((bs_lo >> 24u) | (bs_hi << 8u));
    st[6].y = st[6].y ^ (bs_hi >> 24u);

    // keccak-f[1600], 24 rounds with the inner loops unrolled.
    for (var r: u32 = 0u; r < 24u; r = r + 1u) {
        // theta
        let bc0 = st[0] ^ st[5]  ^ st[10] ^ st[15] ^ st[20];
        let bc1 = st[1] ^ st[6]  ^ st[11] ^ st[16] ^ st[21];
        let bc2 = st[2] ^ st[7]  ^ st[12] ^ st[17] ^ st[22];
        let bc3 = st[3] ^ st[8]  ^ st[13] ^ st[18] ^ st[23];
        let bc4 = st[4] ^ st[9]  ^ st[14] ^ st[19] ^ st[24];

        let d0 = bc4 ^ rotl64(bc1, 1u);
        let d1 = bc0 ^ rotl64(bc2, 1u);
        let d2 = bc1 ^ rotl64(bc3, 1u);
        let d3 = bc2 ^ rotl64(bc4, 1u);
        let d4 = bc3 ^ rotl64(bc0, 1u);

        st[0]  = st[0]  ^ d0; st[1]  = st[1]  ^ d1; st[2]  = st[2]  ^ d2; st[3]  = st[3]  ^ d3; st[4]  = st[4]  ^ d4;
        st[5]  = st[5]  ^ d0; st[6]  = st[6]  ^ d1; st[7]  = st[7]  ^ d2; st[8]  = st[8]  ^ d3; st[9]  = st[9]  ^ d4;
        st[10] = st[10] ^ d0; st[11] = st[11] ^ d1; st[12] = st[12] ^ d2; st[13] = st[13] ^ d3; st[14] = st[14] ^ d4;
        st[15] = st[15] ^ d0; st[16] = st[16] ^ d1; st[17] = st[17] ^ d2; st[18] = st[18] ^ d3; st[19] = st[19] ^ d4;
        st[20] = st[20] ^ d0; st[21] = st[21] ^ d1; st[22] = st[22] ^ d2; st[23] = st[23] ^ d3; st[24] = st[24] ^ d4;

        // rho-pi: chained 24-step lane permutation. PILN/ROTC are baked in.
        var t = st[1];
        var tmp: vec2<u32>;
        tmp = st[10]; st[10] = rotl64(t,  1u); t = tmp;
        tmp = st[ 7]; st[ 7] = rotl64(t,  3u); t = tmp;
        tmp = st[11]; st[11] = rotl64(t,  6u); t = tmp;
        tmp = st[17]; st[17] = rotl64(t, 10u); t = tmp;
        tmp = st[18]; st[18] = rotl64(t, 15u); t = tmp;
        tmp = st[ 3]; st[ 3] = rotl64(t, 21u); t = tmp;
        tmp = st[ 5]; st[ 5] = rotl64(t, 28u); t = tmp;
        tmp = st[16]; st[16] = rotl64(t, 36u); t = tmp;
        tmp = st[ 8]; st[ 8] = rotl64(t, 45u); t = tmp;
        tmp = st[21]; st[21] = rotl64(t, 55u); t = tmp;
        tmp = st[24]; st[24] = rotl64(t,  2u); t = tmp;
        tmp = st[ 4]; st[ 4] = rotl64(t, 14u); t = tmp;
        tmp = st[15]; st[15] = rotl64(t, 27u); t = tmp;
        tmp = st[23]; st[23] = rotl64(t, 41u); t = tmp;
        tmp = st[19]; st[19] = rotl64(t, 56u); t = tmp;
        tmp = st[13]; st[13] = rotl64(t,  8u); t = tmp;
        tmp = st[12]; st[12] = rotl64(t, 25u); t = tmp;
        tmp = st[ 2]; st[ 2] = rotl64(t, 43u); t = tmp;
        tmp = st[20]; st[20] = rotl64(t, 62u); t = tmp;
        tmp = st[14]; st[14] = rotl64(t, 18u); t = tmp;
        tmp = st[22]; st[22] = rotl64(t, 39u); t = tmp;
        tmp = st[ 9]; st[ 9] = rotl64(t, 61u); t = tmp;
        tmp = st[ 6]; st[ 6] = rotl64(t, 20u); t = tmp;
        st[1] = rotl64(t, 44u);

        // chi: each row reads its 5 old lanes, writes its 5 new ones.
        {
            let t0 = st[0]; let t1 = st[1];
            st[0] = st[0] ^ ((~st[1]) & st[2]);
            st[1] = st[1] ^ ((~st[2]) & st[3]);
            st[2] = st[2] ^ ((~st[3]) & st[4]);
            st[3] = st[3] ^ ((~st[4]) & t0);
            st[4] = st[4] ^ ((~t0)    & t1);
        }
        {
            let t0 = st[5]; let t1 = st[6];
            st[5] = st[5] ^ ((~st[6]) & st[7]);
            st[6] = st[6] ^ ((~st[7]) & st[8]);
            st[7] = st[7] ^ ((~st[8]) & st[9]);
            st[8] = st[8] ^ ((~st[9]) & t0);
            st[9] = st[9] ^ ((~t0)    & t1);
        }
        {
            let t0 = st[10]; let t1 = st[11];
            st[10] = st[10] ^ ((~st[11]) & st[12]);
            st[11] = st[11] ^ ((~st[12]) & st[13]);
            st[12] = st[12] ^ ((~st[13]) & st[14]);
            st[13] = st[13] ^ ((~st[14]) & t0);
            st[14] = st[14] ^ ((~t0)     & t1);
        }
        {
            let t0 = st[15]; let t1 = st[16];
            st[15] = st[15] ^ ((~st[16]) & st[17]);
            st[16] = st[16] ^ ((~st[17]) & st[18]);
            st[17] = st[17] ^ ((~st[18]) & st[19]);
            st[18] = st[18] ^ ((~st[19]) & t0);
            st[19] = st[19] ^ ((~t0)     & t1);
        }
        {
            let t0 = st[20]; let t1 = st[21];
            st[20] = st[20] ^ ((~st[21]) & st[22]);
            st[21] = st[21] ^ ((~st[22]) & st[23]);
            st[22] = st[22] ^ ((~st[23]) & st[24]);
            st[23] = st[23] ^ ((~st[24]) & t0);
            st[24] = st[24] ^ ((~t0)     & t1);
        }

        // iota
        st[0] = st[0] ^ RC[r];
    }

    let a1 = st[1];
    let a2 = st[2];
    let a3 = st[3];

    if (all((a1 & mask_lanes[0]) == mask_lanes[1]) &&
        all((a2 & mask_lanes[2]) == mask_lanes[3]) &&
        all((a3 & mask_lanes[4]) == mask_lanes[5])) {
        loop {
            let cas = atomicCompareExchangeWeak(&found, 0u, 1u);
            if (cas.exchanged) {
                result[0] = salt.x;
                result[1] = salt.y;
                result[2] = a1.x;
                result[3] = a1.y;
                result[4] = a2.x;
                result[5] = a2.y;
                result[6] = a3.x;
                result[7] = a3.y;
                break;
            }
            if (cas.old_value != 0u) { break; }
        }
    }
}
