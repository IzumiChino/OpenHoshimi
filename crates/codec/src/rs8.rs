//! Reed-Solomon RS(255, 223) over GF(256) for the SSDV image
//! protocol.
//!
//! Hand-translated from Phil Karn's portable C reference (`rs8.c`),
//! with the SSDV-specific tweaks Philip Heron applied in
//! [`fsphil/ssdv`](https://github.com/fsphil/ssdv) preserved
//! verbatim. The constants below come straight from the original
//! lookup tables; algorithm steps follow the same Berlekamp-Massey,
//! Chien search, and Forney evaluation flow.
//!
//! Field parameters: GF(256) with primitive polynomial
//! `x^8 + x^7 + x^2 + x + 1` (0x187), first consecutive root
//! `alpha^112`, primitive element exponent `11`. These are required
//! by SSDV; do not change them.
//!
//! Decoder accepts an in-place 255-byte block (data + parity, with
//! the SSDV sync byte excluded) and returns the number of corrected
//! errors, or [`RsError::Uncorrectable`] when the syndromes are
//! non-zero but cannot be inverted.
//!
//! Provenance: Karn's RS implementation is LGPL-2.1-or-later, the
//! ssdv tweaks are GPL-3.0-or-later. Both are compatible with this
//! workspace's GPL-3.0-or-later license.

const NN: usize = 255;
const NROOTS: usize = 32;
const FCR: usize = 112;
const PRIM: usize = 11;
const IPRIM: usize = 116;

/// Special reserved value encoding zero in index form.
const A0: u8 = NN as u8;

const ALPHA_TO: [u8; 256] = [
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x87, 0x89, 0x95, 0xAD, 0xDD, 0x3D, 0x7A, 0xF4,
    0x6F, 0xDE, 0x3B, 0x76, 0xEC, 0x5F, 0xBE, 0xFB, 0x71, 0xE2, 0x43, 0x86, 0x8B, 0x91, 0xA5, 0xCD,
    0x1D, 0x3A, 0x74, 0xE8, 0x57, 0xAE, 0xDB, 0x31, 0x62, 0xC4, 0x0F, 0x1E, 0x3C, 0x78, 0xF0, 0x67,
    0xCE, 0x1B, 0x36, 0x6C, 0xD8, 0x37, 0x6E, 0xDC, 0x3F, 0x7E, 0xFC, 0x7F, 0xFE, 0x7B, 0xF6, 0x6B,
    0xD6, 0x2B, 0x56, 0xAC, 0xDF, 0x39, 0x72, 0xE4, 0x4F, 0x9E, 0xBB, 0xF1, 0x65, 0xCA, 0x13, 0x26,
    0x4C, 0x98, 0xB7, 0xE9, 0x55, 0xAA, 0xD3, 0x21, 0x42, 0x84, 0x8F, 0x99, 0xB5, 0xED, 0x5D, 0xBA,
    0xF3, 0x61, 0xC2, 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x07, 0x0E, 0x1C, 0x38, 0x70, 0xE0,
    0x47, 0x8E, 0x9B, 0xB1, 0xE5, 0x4D, 0x9A, 0xB3, 0xE1, 0x45, 0x8A, 0x93, 0xA1, 0xC5, 0x0D, 0x1A,
    0x34, 0x68, 0xD0, 0x27, 0x4E, 0x9C, 0xBF, 0xF9, 0x75, 0xEA, 0x53, 0xA6, 0xCB, 0x11, 0x22, 0x44,
    0x88, 0x97, 0xA9, 0xD5, 0x2D, 0x5A, 0xB4, 0xEF, 0x59, 0xB2, 0xE3, 0x41, 0x82, 0x83, 0x81, 0x85,
    0x8D, 0x9D, 0xBD, 0xFD, 0x7D, 0xFA, 0x73, 0xE6, 0x4B, 0x96, 0xAB, 0xD1, 0x25, 0x4A, 0x94, 0xAF,
    0xD9, 0x35, 0x6A, 0xD4, 0x2F, 0x5E, 0xBC, 0xFF, 0x79, 0xF2, 0x63, 0xC6, 0x0B, 0x16, 0x2C, 0x58,
    0xB0, 0xE7, 0x49, 0x92, 0xA3, 0xC1, 0x05, 0x0A, 0x14, 0x28, 0x50, 0xA0, 0xC7, 0x09, 0x12, 0x24,
    0x48, 0x90, 0xA7, 0xC9, 0x15, 0x2A, 0x54, 0xA8, 0xD7, 0x29, 0x52, 0xA4, 0xCF, 0x19, 0x32, 0x64,
    0xC8, 0x17, 0x2E, 0x5C, 0xB8, 0xF7, 0x69, 0xD2, 0x23, 0x46, 0x8C, 0x9F, 0xB9, 0xF5, 0x6D, 0xDA,
    0x33, 0x66, 0xCC, 0x1F, 0x3E, 0x7C, 0xF8, 0x77, 0xEE, 0x5B, 0xB6, 0xEB, 0x51, 0xA2, 0xC3, 0x00,
];

const INDEX_OF: [u8; 256] = [
    0xFF, 0x00, 0x01, 0x63, 0x02, 0xC6, 0x64, 0x6A, 0x03, 0xCD, 0xC7, 0xBC, 0x65, 0x7E, 0x6B, 0x2A,
    0x04, 0x8D, 0xCE, 0x4E, 0xC8, 0xD4, 0xBD, 0xE1, 0x66, 0xDD, 0x7F, 0x31, 0x6C, 0x20, 0x2B, 0xF3,
    0x05, 0x57, 0x8E, 0xE8, 0xCF, 0xAC, 0x4F, 0x83, 0xC9, 0xD9, 0xD5, 0x41, 0xBE, 0x94, 0xE2, 0xB4,
    0x67, 0x27, 0xDE, 0xF0, 0x80, 0xB1, 0x32, 0x35, 0x6D, 0x45, 0x21, 0x12, 0x2C, 0x0D, 0xF4, 0x38,
    0x06, 0x9B, 0x58, 0x1A, 0x8F, 0x79, 0xE9, 0x70, 0xD0, 0xC2, 0xAD, 0xA8, 0x50, 0x75, 0x84, 0x48,
    0xCA, 0xFC, 0xDA, 0x8A, 0xD6, 0x54, 0x42, 0x24, 0xBF, 0x98, 0x95, 0xF9, 0xE3, 0x5E, 0xB5, 0x15,
    0x68, 0x61, 0x28, 0xBA, 0xDF, 0x4C, 0xF1, 0x2F, 0x81, 0xE6, 0xB2, 0x3F, 0x33, 0xEE, 0x36, 0x10,
    0x6E, 0x18, 0x46, 0xA6, 0x22, 0x88, 0x13, 0xF7, 0x2D, 0xB8, 0x0E, 0x3D, 0xF5, 0xA4, 0x39, 0x3B,
    0x07, 0x9E, 0x9C, 0x9D, 0x59, 0x9F, 0x1B, 0x08, 0x90, 0x09, 0x7A, 0x1C, 0xEA, 0xA0, 0x71, 0x5A,
    0xD1, 0x1D, 0xC3, 0x7B, 0xAE, 0x0A, 0xA9, 0x91, 0x51, 0x5B, 0x76, 0x72, 0x85, 0xA1, 0x49, 0xEB,
    0xCB, 0x7C, 0xFD, 0xC4, 0xDB, 0x1E, 0x8B, 0xD2, 0xD7, 0x92, 0x55, 0xAA, 0x43, 0x0B, 0x25, 0xAF,
    0xC0, 0x73, 0x99, 0x77, 0x96, 0x5C, 0xFA, 0x52, 0xE4, 0xEC, 0x5F, 0x4A, 0xB6, 0xA2, 0x16, 0x86,
    0x69, 0xC5, 0x62, 0xFE, 0x29, 0x7D, 0xBB, 0xCC, 0xE0, 0xD3, 0x4D, 0x8C, 0xF2, 0x1F, 0x30, 0xDC,
    0x82, 0xAB, 0xE7, 0x56, 0xB3, 0x93, 0x40, 0xD8, 0x34, 0xB0, 0xEF, 0x26, 0x37, 0x0C, 0x11, 0x44,
    0x6F, 0x78, 0x19, 0x9A, 0x47, 0x74, 0xA7, 0xC1, 0x23, 0x53, 0x89, 0xFB, 0x14, 0x5D, 0xF8, 0x97,
    0x2E, 0x4B, 0xB9, 0x60, 0x0F, 0xED, 0x3E, 0xE5, 0xF6, 0x87, 0xA5, 0x17, 0x3A, 0xA3, 0x3C, 0xB7,
];

const GENPOLY: [u8; 33] = [
    0x00, 0xF9, 0x3B, 0x42, 0x04, 0x2B, 0x7E, 0xFB, 0x61, 0x1E, 0x03, 0xD5, 0x32, 0x42, 0xAA, 0x05,
    0x18, 0x05, 0xAA, 0x42, 0x32, 0xD5, 0x03, 0x1E, 0x61, 0xFB, 0x7E, 0x2B, 0x04, 0x42, 0x3B, 0xF9,
    0x00,
];

/// Errors returned by [`decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsError {
    /// Input slice was not exactly 255 bytes.
    BadLength,
    /// Syndromes were non-zero but the locator polynomial degree did
    /// not match the Chien-search root count, indicating more than
    /// 16 byte errors.
    Uncorrectable,
}

impl core::fmt::Display for RsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RsError::BadLength => f.write_str("RS block must be 255 bytes"),
            RsError::Uncorrectable => f.write_str("RS block has uncorrectable errors"),
        }
    }
}

impl std::error::Error for RsError {}

fn mod_nn(mut x: i32) -> i32 {
    while x >= NN as i32 {
        x -= NN as i32;
        x = (x >> 8) + (x & 0xFF);
    }
    x
}

/// Encode 223 data bytes into a 255-byte RS(255, 223) codeword.
///
/// `block` must be exactly 255 bytes. Bytes `[0..223]` carry the
/// caller's data and will not be modified; bytes `[223..255]` are
/// overwritten with parity. Used by tests; SSDV downlink only needs
/// [`decode`].
pub fn encode(block: &mut [u8]) -> Result<(), RsError> {
    if block.len() != NN {
        return Err(RsError::BadLength);
    }
    let (data, parity) = block.split_at_mut(NN - NROOTS);
    parity.fill(0);
    for &byte in data.iter() {
        let feedback = INDEX_OF[(byte ^ parity[0]) as usize];
        if feedback != A0 {
            for j in 1..NROOTS {
                parity[j] ^=
                    ALPHA_TO[mod_nn(feedback as i32 + GENPOLY[NROOTS - j] as i32) as usize];
            }
        }
        parity.copy_within(1..NROOTS, 0);
        parity[NROOTS - 1] = if feedback != A0 {
            ALPHA_TO[mod_nn(feedback as i32 + GENPOLY[0] as i32) as usize]
        } else {
            0
        };
    }
    Ok(())
}

/// Decode a 255-byte RS(255, 223) codeword in place.
///
/// On success, returns the number of byte errors that were corrected
/// (zero when the codeword was already clean). The first 223 bytes
/// of `block` carry the recovered data; the trailing 32 parity bytes
/// are left in their post-correction state.
///
/// # Errors
///
/// Returns [`RsError::BadLength`] if the slice is not exactly 255
/// bytes long, or [`RsError::Uncorrectable`] when more than 16 byte
/// errors are present.
pub fn decode(block: &mut [u8]) -> Result<usize, RsError> {
    if block.len() != NN {
        return Err(RsError::BadLength);
    }

    let mut s = [0u8; NROOTS];
    for syn in s.iter_mut() {
        *syn = block[0];
    }
    for &b in &block[1..NN] {
        for (i, syn) in s.iter_mut().enumerate() {
            if *syn == 0 {
                *syn = b;
            } else {
                let exp = mod_nn(INDEX_OF[*syn as usize] as i32 + ((FCR + i) * PRIM) as i32);
                *syn = b ^ ALPHA_TO[exp as usize];
            }
        }
    }

    let mut syn_error: u8 = 0;
    for syn in s.iter_mut() {
        syn_error |= *syn;
        *syn = INDEX_OF[*syn as usize];
    }
    if syn_error == 0 {
        return Ok(0);
    }

    let mut lambda = [0u8; NROOTS + 1];
    lambda[0] = 1;
    let mut b = [0u8; NROOTS + 1];
    for (slot, lam) in b.iter_mut().zip(lambda.iter()) {
        *slot = INDEX_OF[*lam as usize];
    }

    let mut t = [0u8; NROOTS + 1];
    let mut r: usize = 0;
    let mut el: usize = 0;
    while r < NROOTS {
        r += 1;
        let mut discr_r: u8 = 0;
        for i in 0..r {
            if lambda[i] != 0 && s[r - i - 1] != A0 {
                discr_r ^= ALPHA_TO
                    [mod_nn(INDEX_OF[lambda[i] as usize] as i32 + s[r - i - 1] as i32) as usize];
            }
        }
        let discr_idx = INDEX_OF[discr_r as usize];
        if discr_idx == A0 {
            b.copy_within(0..NROOTS, 1);
            b[0] = A0;
        } else {
            t[0] = lambda[0];
            for i in 0..NROOTS {
                t[i + 1] = if b[i] != A0 {
                    lambda[i + 1] ^ ALPHA_TO[mod_nn(discr_idx as i32 + b[i] as i32) as usize]
                } else {
                    lambda[i + 1]
                };
            }
            if 2 * el < r {
                el = r - el;
                for i in 0..=NROOTS {
                    b[i] = if lambda[i] == 0 {
                        A0
                    } else {
                        mod_nn(INDEX_OF[lambda[i] as usize] as i32 - discr_idx as i32 + NN as i32)
                            as u8
                    };
                }
            } else {
                b.copy_within(0..NROOTS, 1);
                b[0] = A0;
            }
            lambda.copy_from_slice(&t);
        }
    }

    let mut deg_lambda = 0usize;
    for (i, lam) in lambda.iter_mut().enumerate() {
        *lam = INDEX_OF[*lam as usize];
        if *lam != A0 {
            deg_lambda = i;
        }
    }

    let mut reg = [0u8; NROOTS + 1];
    reg[1..=NROOTS].copy_from_slice(&lambda[1..=NROOTS]);
    let mut count = 0usize;
    let mut root = [0u8; NROOTS];
    let mut loc = [0u8; NROOTS];
    let mut k_idx: i32 = IPRIM as i32 - 1;
    for i in 1..=NN {
        let mut q: u8 = 1;
        for j in (1..=deg_lambda).rev() {
            if reg[j] != A0 {
                reg[j] = mod_nn(reg[j] as i32 + j as i32) as u8;
                q ^= ALPHA_TO[reg[j] as usize];
            }
        }
        if q == 0 {
            root[count] = i as u8;
            loc[count] = k_idx as u8;
            count += 1;
            if count == deg_lambda {
                break;
            }
        }
        k_idx = mod_nn(k_idx + IPRIM as i32);
    }

    if deg_lambda != count {
        return Err(RsError::Uncorrectable);
    }

    let deg_omega = deg_lambda.saturating_sub(1);
    let mut omega = [0u8; NROOTS + 1];
    for i in 0..=deg_omega {
        let mut tmp: u8 = 0;
        for j in (0..=i).rev() {
            if s[i - j] != A0 && lambda[j] != A0 {
                tmp ^= ALPHA_TO[mod_nn(s[i - j] as i32 + lambda[j] as i32) as usize];
            }
        }
        omega[i] = INDEX_OF[tmp as usize];
    }

    for j in (0..count).rev() {
        let mut num1: u8 = 0;
        for i in (0..=deg_omega).rev() {
            if omega[i] != A0 {
                num1 ^= ALPHA_TO[mod_nn(omega[i] as i32 + (i as i32 * root[j] as i32)) as usize];
            }
        }
        let num2 = ALPHA_TO[mod_nn(root[j] as i32 * (FCR as i32 - 1) + NN as i32) as usize];
        let mut den: u8 = 0;
        let start = (deg_lambda.min(NROOTS - 1)) & !1usize;
        let mut i = start as i32;
        while i >= 0 {
            if lambda[(i + 1) as usize] != A0 {
                den ^=
                    ALPHA_TO[mod_nn(lambda[(i + 1) as usize] as i32 + i * root[j] as i32) as usize];
            }
            i -= 2;
        }
        if num1 != 0 && (loc[j] as usize) < NN {
            block[loc[j] as usize] ^= ALPHA_TO[mod_nn(
                INDEX_OF[num1 as usize] as i32 + INDEX_OF[num2 as usize] as i32 + NN as i32
                    - INDEX_OF[den as usize] as i32,
            ) as usize];
        }
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_codeword(seed: u8) -> [u8; NN] {
        let mut block = [0u8; NN];
        for (i, byte) in block[..NN - NROOTS].iter_mut().enumerate() {
            *byte = seed.wrapping_add(i as u8).wrapping_mul(31);
        }
        encode(&mut block).expect("encode");
        block
    }

    #[test]
    fn clean_codeword_decodes_to_zero_errors() {
        let mut block = make_codeword(0xA5);
        assert_eq!(decode(&mut block), Ok(0));
    }

    #[test]
    fn corrects_up_to_sixteen_errors() {
        let original = make_codeword(0x42);
        let mut block = original;
        let positions = [
            3usize, 17, 41, 67, 89, 105, 130, 150, 170, 188, 200, 210, 220, 230, 240, 252,
        ];
        assert_eq!(positions.len(), 16);
        for &p in positions.iter() {
            block[p] ^= 0xA7;
        }
        let corrected = decode(&mut block).expect("decode");
        assert_eq!(corrected, 16);
        assert_eq!(block, original);
    }

    #[test]
    fn seventeen_errors_are_uncorrectable() {
        let mut block = make_codeword(0x99);
        let positions = [
            1usize, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65,
        ];
        for &p in positions.iter() {
            block[p] ^= 0xFF;
        }
        // RS(255,223) cannot correct 17 errors; expect either an
        // explicit Uncorrectable result or a miscorrection. Either
        // way, the decoder must not silently treat the block as
        // clean (the original constants would then have been wrong).
        match decode(&mut block) {
            Err(RsError::Uncorrectable) => {}
            Ok(n) => assert!(
                n > 0,
                "decoder must report errors when syndromes are non-zero"
            ),
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_length() {
        let mut buf = [0u8; 100];
        assert_eq!(decode(&mut buf), Err(RsError::BadLength));
    }
}
