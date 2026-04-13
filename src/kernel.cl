// Keccak-256 + CREATE2 address match, one salt per thread.
//
// The host precomputes a 136-byte keccak input with create2Salt = argsHash
// (i.e. the salt=0 baseline), packs it into 17 little-endian u64 "lanes",
// and passes it as base_state. Each thread then XORs a u64 salt counter into
// the bytes of create2Salt that correspond to Solidity's `bytes32(uint256(salt))`
// — the last 8 bytes of create2Salt, which straddle lanes 5 and 6 — runs a
// single Keccak-f[1600], and checks whether the resulting address matches a
// caller-supplied mask/match on lanes 1..3 of the squeezed state.

__constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

__constant uchar PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

__constant uchar ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

static inline ulong rotl64(ulong x, uint n) {
    return (x << n) | (x >> (64 - n));
}

static void keccakf(ulong st[25]) {
    ulong bc[5];
    for (int r = 0; r < 24; r++) {
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        for (int i = 0; i < 5; i++) {
            ulong t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            st[i]      ^= t;
            st[i + 5]  ^= t;
            st[i + 10] ^= t;
            st[i + 15] ^= t;
            st[i + 20] ^= t;
        }

        ulong t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = PILN[i];
            ulong tmp = st[j];
            st[j] = rotl64(t, ROTC[i]);
            t = tmp;
        }

        for (int j = 0; j < 25; j += 5) {
            ulong t0 = st[j];
            ulong t1 = st[j + 1];
            st[j]     ^= (~st[j + 1]) & st[j + 2];
            st[j + 1] ^= (~st[j + 2]) & st[j + 3];
            st[j + 2] ^= (~st[j + 3]) & st[j + 4];
            st[j + 3] ^= (~st[j + 4]) & t0;
            st[j + 4] ^= (~t0) & t1;
        }

        st[0] ^= RC[r];
    }
}

static inline ulong bswap64(ulong x) {
    return ((x & 0x00000000000000FFUL) << 56) |
           ((x & 0x000000000000FF00UL) << 40) |
           ((x & 0x0000000000FF0000UL) << 24) |
           ((x & 0x00000000FF000000UL) <<  8) |
           ((x & 0x000000FF00000000UL) >>  8) |
           ((x & 0x0000FF0000000000UL) >> 24) |
           ((x & 0x00FF000000000000UL) >> 40) |
           ((x & 0xFF00000000000000UL) >> 56);
}

__kernel void mine(
    ulong start_salt,
    ulong stride,
    ulong max_salt,
    __constant ulong* base_state,
    __constant ulong* mask_lanes,
    volatile __global int* found,
    __global ulong* result
) {
    ulong i = (ulong) get_global_id(0);
    ulong salt = start_salt + i * stride;
    if (salt >= max_salt) return;

    ulong st[25];
    st[0]  = base_state[0];
    st[1]  = base_state[1];
    st[2]  = base_state[2];
    st[3]  = base_state[3];
    st[4]  = base_state[4];
    st[5]  = base_state[5];
    st[6]  = base_state[6];
    st[7]  = base_state[7];
    st[8]  = base_state[8];
    st[9]  = base_state[9];
    st[10] = base_state[10];
    st[11] = base_state[11];
    st[12] = base_state[12];
    st[13] = base_state[13];
    st[14] = base_state[14];
    st[15] = base_state[15];
    st[16] = base_state[16];
    st[17] = 0;
    st[18] = 0;
    st[19] = 0;
    st[20] = 0;
    st[21] = 0;
    st[22] = 0;
    st[23] = 0;
    st[24] = 0;

    // The u64 salt is written as bytes[24..32] of create2Salt, big-endian
    // (matching Solidity's `bytes32(uint256(salt))`). Those 8 bytes land at
    // message bytes 45..53, which span lane 5 bytes 5..7 and lane 6 bytes 0..4.
    ulong bs = bswap64(salt);
    st[5] ^= (bs & 0xFFFFFFUL) << 40;
    st[6] ^= bs >> 24;

    keccakf(st);

    // The 160-bit address is bytes 12..32 of the squeezed state:
    //   addr[0..4]   = lane 1 bytes 4..7 (bits 32..63 of st[1])
    //   addr[4..12]  = lane 2 bytes 0..7 (all of st[2])
    //   addr[12..20] = lane 3 bytes 0..7 (all of st[3])
    ulong a1 = st[1];
    ulong a2 = st[2];
    ulong a3 = st[3];

    if (((a1 & mask_lanes[0]) == mask_lanes[1]) &&
        ((a2 & mask_lanes[2]) == mask_lanes[3]) &&
        ((a3 & mask_lanes[4]) == mask_lanes[5])) {
        if (atomic_cmpxchg(found, 0, 1) == 0) {
            result[0] = salt;
            result[1] = a1;
            result[2] = a2;
            result[3] = a3;
        }
    }
}
