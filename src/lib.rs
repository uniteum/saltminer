//! Pure-Rust helpers shared by the saltminer binary and its tests.
//!
//! Everything here is OpenCL-free so `cargo test --lib` can exercise the
//! tricky lane-packing and address-extraction math without needing the
//! OpenCL loader installed.

use anyhow::{bail, Context, Result};

pub fn parse_hex_bytes<const N: usize>(s: &str) -> Result<[u8; N]> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(s).with_context(|| format!("invalid hex: {s}"))?;
    if bytes.len() != N {
        bail!("expected {} bytes of hex, got {}", N, bytes.len());
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&bytes);
    Ok(out)
}

pub fn parse_u64(s: &str) -> Result<u64> {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).with_context(|| format!("invalid hex u64: {s}"))
    } else {
        s.parse::<u64>().with_context(|| format!("invalid decimal u64: {s}"))
    }
}

pub fn parse_shard(s: &str) -> Result<(u64, u64)> {
    let (w, n) = s.split_once('/').context("shard must be w/N")?;
    let w: u64 = w.parse().context("shard worker index not a number")?;
    let n: u64 = n.parse().context("shard worker count not a number")?;
    if n == 0 {
        bail!("shard worker count must be > 0");
    }
    if w >= n {
        bail!("shard worker index must be < count");
    }
    Ok((w, n))
}

/// Pack the 85-byte CREATE2 keccak input into 17 little-endian u64 lanes,
/// with `create2Salt = argsHash` (the salt=0 baseline). The kernel XORs the
/// salt contribution into lanes 5 and 6 at dispatch time.
pub fn compute_base_state(
    deployer: &[u8; 20],
    args_hash: &[u8; 32],
    initcode_hash: &[u8; 32],
) -> [u64; 17] {
    let mut msg = [0u8; 136];
    msg[0] = 0xff;
    msg[1..21].copy_from_slice(deployer);
    msg[21..53].copy_from_slice(args_hash);
    msg[53..85].copy_from_slice(initcode_hash);
    msg[85] = 0x01;
    msg[135] = 0x80;
    let mut lanes = [0u64; 17];
    for (i, lane) in lanes.iter_mut().enumerate() {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&msg[i * 8..i * 8 + 8]);
        *lane = u64::from_le_bytes(buf);
    }
    lanes
}

/// Translate a 20-byte address mask/target pair into the three state lanes the
/// kernel compares against. The 160-bit address occupies:
///   addr[0..4]   = bits 32..63 of squeezed state lane 1
///   addr[4..12]  = all of state lane 2 (little-endian)
///   addr[12..20] = all of state lane 3 (little-endian)
pub fn compute_lane_masks(mask: &[u8; 20], target: &[u8; 20]) -> [u64; 6] {
    let l1_mask = ((mask[0] as u64) << 32)
        | ((mask[1] as u64) << 40)
        | ((mask[2] as u64) << 48)
        | ((mask[3] as u64) << 56);
    let l1_target = ((target[0] as u64) << 32)
        | ((target[1] as u64) << 40)
        | ((target[2] as u64) << 48)
        | ((target[3] as u64) << 56);
    let l2_mask = u64::from_le_bytes(mask[4..12].try_into().unwrap());
    let l2_target = u64::from_le_bytes(target[4..12].try_into().unwrap());
    let l3_mask = u64::from_le_bytes(mask[12..20].try_into().unwrap());
    let l3_target = u64::from_le_bytes(target[12..20].try_into().unwrap());
    [l1_mask, l1_target, l2_mask, l2_target, l3_mask, l3_target]
}

pub fn address_from_state(a1: u64, a2: u64, a3: u64) -> [u8; 20] {
    let mut out = [0u8; 20];
    out[0] = (a1 >> 32) as u8;
    out[1] = (a1 >> 40) as u8;
    out[2] = (a1 >> 48) as u8;
    out[3] = (a1 >> 56) as u8;
    out[4..12].copy_from_slice(&a2.to_le_bytes());
    out[12..20].copy_from_slice(&a3.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiny_keccak::{Hasher, Keccak};

    fn fixed_inputs() -> ([u8; 20], [u8; 32], [u8; 32]) {
        let deployer =
            parse_hex_bytes::<20>("0x14ae57aed6ac1cd48fa811ed885ab4a4c5e28c42").unwrap();
        let args_hash = parse_hex_bytes::<32>(
            "0x0123456789abcdeffedcba98765432100f1e2d3c4b5a69788796a5b4c3d2e1f0",
        )
        .unwrap();
        let initcode_hash = parse_hex_bytes::<32>(
            "0xdeadbeefcafebabef00dfacefeedc0de1234567890abcdefaabbccddeeff0011",
        )
        .unwrap();
        (deployer, args_hash, initcode_hash)
    }

    /// Manually build the 85-byte CREATE2 input plus keccak padding, i.e.
    /// `0xff ‖ deployer ‖ (argsHash XOR bytes32(uint256(salt))) ‖ initcodeHash`
    /// padded to 136 bytes. The kernel's lane-XOR must reproduce this byte for byte.
    fn expected_message_bytes(
        deployer: &[u8; 20],
        args_hash: &[u8; 32],
        initcode_hash: &[u8; 32],
        salt: u64,
    ) -> [u8; 136] {
        let mut create2_salt = *args_hash;
        let salt_be = salt.to_be_bytes();
        for i in 0..8 {
            create2_salt[24 + i] ^= salt_be[i];
        }
        let mut msg = [0u8; 136];
        msg[0] = 0xff;
        msg[1..21].copy_from_slice(deployer);
        msg[21..53].copy_from_slice(&create2_salt);
        msg[53..85].copy_from_slice(initcode_hash);
        msg[85] = 0x01;
        msg[135] = 0x80;
        msg
    }

    /// Apply the kernel's lane XOR on top of the host-precomputed base_state,
    /// mirroring exactly what src/kernel.cl does:
    ///     bs = bswap64(salt)
    ///     st[5] ^= (bs & 0xFFFFFF) << 40
    ///     st[6] ^= bs >> 24
    fn kernel_lanes_for(base_state: [u64; 17], salt: u64) -> [u64; 17] {
        let mut st = base_state;
        let bs = salt.swap_bytes();
        st[5] ^= (bs & 0xFFFFFF) << 40;
        st[6] ^= bs >> 24;
        st
    }

    #[test]
    fn base_state_plus_kernel_xor_matches_bytewise_message() {
        let (deployer, args_hash, initcode_hash) = fixed_inputs();
        for &salt in &[
            0u64,
            1,
            0xff,
            0x100,
            0xdeadbeef,
            0x0123456789abcdef,
            u64::MAX,
        ] {
            let base = compute_base_state(&deployer, &args_hash, &initcode_hash);
            let lanes = kernel_lanes_for(base, salt);
            let mut got = [0u8; 136];
            for (i, lane) in lanes.iter().enumerate() {
                got[i * 8..i * 8 + 8].copy_from_slice(&lane.to_le_bytes());
            }
            let want = expected_message_bytes(&deployer, &args_hash, &initcode_hash, salt);
            assert_eq!(got, want, "salt = {salt:#x}");
        }
    }

    #[test]
    fn address_from_state_matches_keccak_bytes_12_32() {
        let (deployer, args_hash, initcode_hash) = fixed_inputs();
        for &salt in &[0u64, 42, 0xabcdef, u64::MAX] {
            let msg = expected_message_bytes(&deployer, &args_hash, &initcode_hash, salt);
            // tiny-keccak applies its own padding; feed the 85 logical bytes.
            let mut hasher = Keccak::v256();
            hasher.update(&msg[..85]);
            let mut out = [0u8; 32];
            hasher.finalize(&mut out);

            let a1 = u64::from_le_bytes(out[8..16].try_into().unwrap());
            let a2 = u64::from_le_bytes(out[16..24].try_into().unwrap());
            let a3 = u64::from_le_bytes(out[24..32].try_into().unwrap());
            let got = address_from_state(a1, a2, a3);
            let want: [u8; 20] = out[12..32].try_into().unwrap();
            assert_eq!(got, want, "salt = {salt:#x}");
        }
    }

    #[test]
    fn lane_masks_pack_correctly() {
        let mask = [
            0xff, 0xff, 0xff, 0xff, // addr[0..4]
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr[4..12]
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, // addr[12..20]
        ];
        let m = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0xde, 0xad,
        ];
        let lanes = compute_lane_masks(&mask, &m);
        // Lane 1 mask has its high 32 bits set; low 32 bits zero (address bytes
        // 0..3 live in bits 32..63 of state lane 1).
        assert_eq!(lanes[0], 0xffff_ffff_0000_0000);
        assert_eq!(lanes[1], 0x0000_0000_0000_0000);
        // Lane 2 covers address bytes 4..11, all zero here.
        assert_eq!(lanes[2], 0);
        assert_eq!(lanes[3], 0);
        // Lane 3 covers address bytes 12..19. Bytes 18..19 land in bits 48..63
        // of the lane, little-endian — 0xde at bit 48..55, 0xad at bit 56..63.
        assert_eq!(lanes[4], 0xffff_0000_0000_0000);
        assert_eq!(lanes[5], 0xadde_0000_0000_0000);
    }
}
