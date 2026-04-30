// WebGPU port of src/kernel.cl. Each invocation tries one salt:
// reconstruct the 136-byte CREATE2 keccak input from the host-precomputed
// 17-lane base_state, XOR the salt into lanes 5/6 the same way the OpenCL
// kernel does, run keccak-f[1600], and check the address lanes against the
// caller's mask/target. WGSL has no u64, so every lane is a vec2<u32> with
// .x = low 32 bits and .y = high 32 bits.

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

const PILN: array<u32, 24> = array<u32, 24>(
    10u, 7u, 11u, 17u, 18u, 3u, 5u, 16u, 8u, 21u, 24u, 4u,
    15u, 23u, 19u, 13u, 12u, 2u, 20u, 14u, 22u, 9u, 6u, 1u,
);

const ROTC: array<u32, 24> = array<u32, 24>(
    1u, 3u, 6u, 10u, 15u, 21u, 28u, 36u, 45u, 55u, 2u, 14u,
    27u, 41u, 56u, 8u, 25u, 43u, 62u, 18u, 39u, 61u, 20u, 44u,
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

@compute @workgroup_size(64)
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
    // (bs & 0xFFFFFF) << 40: low 24 bits live in bs_lo; <<40 lands them at
    // bit positions 40..63, i.e. the high u32 only, shifted left by 8.
    st[5].y = st[5].y ^ ((bs_lo & 0x00FFFFFFu) << 8u);
    // bs >> 24: shift the full 64-bit value right by 24.
    st[6].x = st[6].x ^ ((bs_lo >> 24u) | (bs_hi << 8u));
    st[6].y = st[6].y ^ (bs_hi >> 24u);

    // keccak-f[1600]
    var bc: array<vec2<u32>, 5>;
    for (var r: u32 = 0u; r < 24u; r = r + 1u) {
        // theta
        for (var c: u32 = 0u; c < 5u; c = c + 1u) {
            bc[c] = st[c] ^ st[c + 5u] ^ st[c + 10u] ^ st[c + 15u] ^ st[c + 20u];
        }
        for (var c: u32 = 0u; c < 5u; c = c + 1u) {
            let t = bc[(c + 4u) % 5u] ^ rotl64(bc[(c + 1u) % 5u], 1u);
            st[c]       = st[c] ^ t;
            st[c + 5u]  = st[c + 5u] ^ t;
            st[c + 10u] = st[c + 10u] ^ t;
            st[c + 15u] = st[c + 15u] ^ t;
            st[c + 20u] = st[c + 20u] ^ t;
        }
        // rho-pi
        var t = st[1];
        for (var c: u32 = 0u; c < 24u; c = c + 1u) {
            let j = PILN[c];
            let tmp = st[j];
            st[j] = rotl64(t, ROTC[c]);
            t = tmp;
        }
        // chi
        for (var jbase: u32 = 0u; jbase < 25u; jbase = jbase + 5u) {
            let t0 = st[jbase];
            let t1 = st[jbase + 1u];
            st[jbase]      = st[jbase]      ^ ((~st[jbase + 1u]) & st[jbase + 2u]);
            st[jbase + 1u] = st[jbase + 1u] ^ ((~st[jbase + 2u]) & st[jbase + 3u]);
            st[jbase + 2u] = st[jbase + 2u] ^ ((~st[jbase + 3u]) & st[jbase + 4u]);
            st[jbase + 3u] = st[jbase + 3u] ^ ((~st[jbase + 4u]) & t0);
            st[jbase + 4u] = st[jbase + 4u] ^ ((~t0) & t1);
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
        // Race to claim the result slot. Loop on the weak CAS so a spurious
        // failure doesn't drop a real match.
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
