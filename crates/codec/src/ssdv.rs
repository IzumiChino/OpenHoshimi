//! SSDV packet decoder.
//!
//! SSDV is the slow-scan digital-video protocol developed by Philip
//! Heron for amateur high-altitude balloon and cubesat image
//! downlinks. Each packet is a fixed 256 bytes containing a
//! self-describing header, a JPEG MCU payload, a CRC32, and (for
//! the FEC variant) 32 bytes of Reed-Solomon parity. See
//! [fsphil/ssdv](https://github.com/fsphil/ssdv) (GPL-3.0-or-later)
//! for the reference C implementation.
//!
//! This module decodes a single packet's bytes into a
//! [`SsdvPacket`]; image reassembly across many packets lives in
//! [`crate::image`].

use crate::rs8;

/// Total wire size of an SSDV packet in bytes.
pub const SSDV_PKT_SIZE: usize = 256;
/// Size of the SSDV packet header (sync byte + type + image fields).
pub const SSDV_HEADER_LEN: usize = 0x0F;
/// CRC32 trailer size in bytes.
pub const SSDV_CRC_LEN: usize = 0x04;
/// Reed-Solomon parity size in bytes (FEC variant only).
pub const SSDV_RS_PARITY_LEN: usize = 0x20;
/// Maximum callsign length in characters (base-40 encoded into 32 bits).
pub const SSDV_MAX_CALLSIGN: usize = 6;

/// Whether a packet carries Reed-Solomon parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsdvPacketKind {
    /// Type byte `0x66`. 205-byte payload, 32 bytes of RS(255, 223)
    /// parity covering everything between the sync byte and the
    /// parity field. Recovers up to 16 byte errors per packet.
    WithFec,
    /// Type byte `0x67`. 237-byte payload, no FEC.
    NoFec,
}

impl SsdvPacketKind {
    /// Payload size in bytes for this packet kind.
    pub fn payload_len(self) -> usize {
        match self {
            SsdvPacketKind::WithFec => {
                SSDV_PKT_SIZE - SSDV_HEADER_LEN - SSDV_CRC_LEN - SSDV_RS_PARITY_LEN
            }
            SsdvPacketKind::NoFec => SSDV_PKT_SIZE - SSDV_HEADER_LEN - SSDV_CRC_LEN,
        }
    }
}

/// One decoded SSDV packet.
#[derive(Debug, Clone)]
pub struct SsdvPacket {
    /// FEC-or-not classification.
    pub kind: SsdvPacketKind,
    /// Decoded base-40 callsign. Empty when the encoded value was 0
    /// or out of range.
    pub callsign: String,
    /// Image id within the sender's session (rolls 0..=255).
    pub image_id: u8,
    /// Packet id, 0-based, monotonically increasing within an image.
    pub packet_id: u16,
    /// Image width in pixels (`packet[9] << 4`).
    pub width: u16,
    /// Image height in pixels (`packet[10] << 4`).
    pub height: u16,
    /// `true` when this packet carries the JPEG end-of-image marker
    /// (flags byte bit 2).
    pub eoi: bool,
    /// JPEG quality level 0..=7. Decoded as `((flags >> 3) & 7) ^ 4`
    /// per the reference implementation.
    pub quality: u8,
    /// MCU sampling mode: 0 = 2x2, 1 = 2x1, 2 = 1x2, 3 = 1x1.
    pub mcu_mode: u8,
    /// Bit offset within the payload where the next fresh MCU
    /// begins, or `0xFF` when none.
    pub mcu_offset: u8,
    /// MCU id of the first complete MCU in this packet, or `0xFFFF`
    /// when none.
    pub mcu_id: u16,
    /// Total MCU count for the whole image, computed from
    /// `width * height * mcu_mode_factor`.
    pub mcu_count: u32,
    /// Payload bytes carrying the JPEG MCU stream slice.
    pub payload: Vec<u8>,
    /// Number of byte errors corrected by Reed-Solomon. `0` for a
    /// pristine FEC packet, `None` for `NoFec`, `Some(n)` for FEC
    /// when `n` errors were corrected (or `Some(-1)` would have
    /// meant uncorrectable, but uncorrectable packets are surfaced
    /// as [`SsdvDecodeError::Uncorrectable`] instead of returning a
    /// successful packet).
    pub rs_errors: Option<u32>,
    /// `true` if the CRC32 over `[type .. CRC]` matched.
    pub crc_ok: bool,
    /// Full corrected 256-byte packet bytes, including sync, header,
    /// payload, CRC, and parity (when present). Useful for callers
    /// that want to forward the exact corrected wire bytes onward.
    pub raw: Vec<u8>,
}

/// Errors returned by [`SsdvDecoder::decode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsdvDecodeError {
    /// Input was not exactly 256 bytes.
    BadLength,
    /// Reed-Solomon could not correct the packet to a CRC-clean
    /// state. Covers both genuinely uncorrectable FEC packets and
    /// any packet whose type byte was corrupted to something other
    /// than `0x66`/`0x67` and whose body could not be rescued.
    Uncorrectable,
    /// CRC32 did not match either before or after RS correction.
    BadCrc,
    /// Header `width` or `height` field was zero.
    ZeroDimensions,
    /// Header `mcu_id`/`mcu_offset` was inconsistent with the
    /// computed MCU count or payload size.
    BadMcu,
}

impl core::fmt::Display for SsdvDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SsdvDecodeError::BadLength => f.write_str("packet must be 256 bytes"),
            SsdvDecodeError::Uncorrectable => f.write_str("Reed-Solomon could not correct packet"),
            SsdvDecodeError::BadCrc => f.write_str("CRC32 mismatch"),
            SsdvDecodeError::ZeroDimensions => f.write_str("width or height is zero"),
            SsdvDecodeError::BadMcu => f.write_str("mcu_id/mcu_offset inconsistent with MCU count"),
        }
    }
}

impl std::error::Error for SsdvDecodeError {}

/// Stateless SSDV packet decoder.
#[derive(Debug, Default, Clone, Copy)]
pub struct SsdvDecoder;

impl SsdvDecoder {
    /// Construct a new decoder. Stateless; cheap.
    pub fn new() -> Self {
        Self
    }

    /// Decode one 256-byte SSDV packet.
    ///
    /// Mirrors fsphil's `ssdv_dec_is_packet` (ssdv.c:1429-1530)
    /// three-tier validator:
    ///
    /// 1. If `pkt[1] == 0x67`, try the NoFEC CRC. Hit → done.
    /// 2. If `pkt[1] == 0x66`, try the FEC CRC. Hit → done.
    /// 3. Otherwise (or if 2 failed), force `pkt[1] = 0x66`, run
    ///    Reed-Solomon over `pkt[1..256]`, then re-check the FEC
    ///    CRC. This covers both genuinely corrupted FEC packets and
    ///    packets where the type byte itself flipped — the type
    ///    byte is inside the RS coverage, so a single-bit type
    ///    flip is recoverable.
    ///
    /// Header geometry (`width`, `height`, `mcu_id`, `mcu_offset`)
    /// is sanity-checked against the computed MCU count.
    ///
    /// # Errors
    ///
    /// Returns the appropriate [`SsdvDecodeError`] when the packet
    /// fails any of the validation steps.
    pub fn decode(&self, raw: &[u8]) -> Result<SsdvPacket, SsdvDecodeError> {
        if raw.len() != SSDV_PKT_SIZE {
            return Err(SsdvDecodeError::BadLength);
        }
        let mut pkt = [0u8; SSDV_PKT_SIZE];
        pkt.copy_from_slice(raw);
        pkt[0] = 0x55;

        // Tier 1 + 2: trust the type byte and try a quick CRC.
        if pkt[1] == 0x67 && crc_matches(&pkt, SsdvPacketKind::NoFec) {
            return finish_packet(&pkt, SsdvPacketKind::NoFec, None);
        }
        if pkt[1] == 0x66 && crc_matches(&pkt, SsdvPacketKind::WithFec) {
            return finish_packet(&pkt, SsdvPacketKind::WithFec, Some(0));
        }

        // Tier 3: force FEC, run RS, re-check CRC. fsphil sets
        // pkt[1] = 0x66 unconditionally here (ssdv.c:1491) because
        // the RS field covers the type byte, so a flipped type
        // byte is corrected as a side effect of the RS pass. We do
        // the same so a single-bit type-byte flip never produces a
        // dropped packet on its own.
        pkt[1] = 0x66;
        let corrected =
            rs8::decode(&mut pkt[1..1 + 255]).map_err(|_| SsdvDecodeError::Uncorrectable)?;
        if !crc_matches(&pkt, SsdvPacketKind::WithFec) {
            return Err(SsdvDecodeError::BadCrc);
        }
        finish_packet(&pkt, SsdvPacketKind::WithFec, Some(corrected as u32))
    }
}

fn finish_packet(
    pkt: &[u8; SSDV_PKT_SIZE],
    kind: SsdvPacketKind,
    rs_errors: Option<u32>,
) -> Result<SsdvPacket, SsdvDecodeError> {
    let header = parse_header(pkt);
    if header.width == 0 || header.height == 0 {
        return Err(SsdvDecodeError::ZeroDimensions);
    }
    if header.mcu_id != 0xFFFF {
        if (header.mcu_id as u32) >= header.mcu_count {
            return Err(SsdvDecodeError::BadMcu);
        }
        if (header.mcu_offset as usize) >= kind.payload_len() {
            return Err(SsdvDecodeError::BadMcu);
        }
    }
    let payload_start = SSDV_HEADER_LEN;
    let payload_end = payload_start + kind.payload_len();
    let payload = pkt[payload_start..payload_end].to_vec();
    Ok(SsdvPacket {
        kind,
        callsign: header.callsign,
        image_id: header.image_id,
        packet_id: header.packet_id,
        width: header.width,
        height: header.height,
        eoi: header.eoi,
        quality: header.quality,
        mcu_mode: header.mcu_mode,
        mcu_offset: header.mcu_offset,
        mcu_id: header.mcu_id,
        mcu_count: header.mcu_count,
        payload,
        rs_errors,
        crc_ok: true,
        raw: pkt.to_vec(),
    })
}

struct Header {
    callsign: String,
    image_id: u8,
    packet_id: u16,
    width: u16,
    height: u16,
    eoi: bool,
    quality: u8,
    mcu_mode: u8,
    mcu_offset: u8,
    mcu_id: u16,
    mcu_count: u32,
}

fn parse_header(pkt: &[u8; SSDV_PKT_SIZE]) -> Header {
    let callsign_code = u32::from_be_bytes([pkt[2], pkt[3], pkt[4], pkt[5]]);
    let image_id = pkt[6];
    let packet_id = u16::from_be_bytes([pkt[7], pkt[8]]);
    let width = (pkt[9] as u16) << 4;
    let height = (pkt[10] as u16) << 4;
    let flags = pkt[11];
    let eoi = (flags >> 2) & 1 != 0;
    let quality = ((flags >> 3) & 7) ^ 4;
    let mcu_mode = flags & 0x03;
    let mcu_offset = pkt[12];
    let mcu_id = u16::from_be_bytes([pkt[13], pkt[14]]);
    let mut mcu_count = (pkt[9] as u32) * (pkt[10] as u32);
    match mcu_mode {
        1 | 2 => mcu_count *= 2,
        3 => mcu_count *= 4,
        _ => {}
    }
    Header {
        callsign: decode_callsign(callsign_code),
        image_id,
        packet_id,
        width,
        height,
        eoi,
        quality,
        mcu_mode,
        mcu_offset,
        mcu_id,
        mcu_count,
    }
}

fn crc_matches(pkt: &[u8; SSDV_PKT_SIZE], kind: SsdvPacketKind) -> bool {
    let payload_len = kind.payload_len();
    let crcdata_len = SSDV_HEADER_LEN + payload_len - 1;
    let computed = crc32(&pkt[1..1 + crcdata_len]);
    let i = 1 + crcdata_len;
    let stored = u32::from_be_bytes([pkt[i], pkt[i + 1], pkt[i + 2], pkt[i + 3]]);
    computed == stored
}

/// CRC32 with reflected polynomial 0xEDB88320, init/xorout
/// 0xFFFFFFFF, matching `ssdv.c`'s `crc32` byte-for-byte.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        let mut x: u32 = (crc ^ b as u32) & 0xFF;
        for _ in 0..8 {
            if x & 1 != 0 {
                x = (x >> 1) ^ 0xEDB8_8320;
            } else {
                x >>= 1;
            }
        }
        crc = (crc >> 8) ^ x;
    }
    crc ^ 0xFFFF_FFFF
}

#[doc(hidden)]
#[cfg(test)]
pub(crate) mod tests_helpers {
    /// Re-export of the private CRC32 helper for sibling-module
    /// tests in `image.rs`.
    pub fn crc32(data: &[u8]) -> u32 {
        super::crc32(data)
    }
}

/// Decode the 32-bit base-40 callsign exactly as `decode_callsign`
/// in the SSDV C reference.
pub fn decode_callsign(mut code: u32) -> String {
    if code > 0xF423_FFFF {
        return String::new();
    }
    let mut out = String::new();
    while code != 0 {
        let s = (code % 40) as u8;
        let ch = match s {
            0 => '-',
            1..=10 => (b'0' + s - 1) as char,
            11..=13 => '-',
            _ => (b'A' + s - 14) as char,
        };
        out.push(ch);
        code /= 40;
    }
    out
}

/// Encode a callsign string into a 32-bit base-40 code, matching
/// `encode_callsign` from the SSDV C reference. Used by tests.
pub fn encode_callsign(callsign: &str) -> u32 {
    let bytes: Vec<u8> = callsign.bytes().take(SSDV_MAX_CALLSIGN).collect();
    let mut x: u32 = 0;
    for &c in bytes.iter().rev() {
        x = x.wrapping_mul(40);
        if c.is_ascii_uppercase() {
            x = x.wrapping_add((c - b'A' + 14) as u32);
        } else if c.is_ascii_lowercase() {
            x = x.wrapping_add((c - b'a' + 14) as u32);
        } else if c.is_ascii_digit() {
            x = x.wrapping_add((c - b'0' + 1) as u32);
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_packet(kind: SsdvPacketKind, payload_seed: u8) -> [u8; SSDV_PKT_SIZE] {
        let mut pkt = [0u8; SSDV_PKT_SIZE];
        pkt[0] = 0x55;
        pkt[1] = match kind {
            SsdvPacketKind::WithFec => 0x66,
            SsdvPacketKind::NoFec => 0x67,
        };
        let cs = encode_callsign("HSCAT1");
        pkt[2] = (cs >> 24) as u8;
        pkt[3] = (cs >> 16) as u8;
        pkt[4] = (cs >> 8) as u8;
        pkt[5] = cs as u8;
        pkt[6] = 0x42; // image id
        pkt[7] = 0x00; // packet id high
        pkt[8] = 0x05; // packet id low = 5
        pkt[9] = 20; // width = 320 px = 20 << 4
        pkt[10] = 15; // height = 240 px = 15 << 4
                      // flags byte: ((quality ^ 4) << 3) | (eoi << 2) | mcu_mode.
                      // For quality=4, mcu_mode=0, eoi=0 the byte is 0.
        pkt[11] = 0;
        pkt[12] = 0; // mcu offset
        pkt[13] = 0; // mcu id high
        pkt[14] = 5; // mcu id low
        let payload_len = kind.payload_len();
        for (i, byte) in pkt[SSDV_HEADER_LEN..SSDV_HEADER_LEN + payload_len]
            .iter_mut()
            .enumerate()
        {
            *byte = payload_seed.wrapping_add(i as u8);
        }
        let crcdata_len = SSDV_HEADER_LEN + payload_len - 1;
        let crc = crc32(&pkt[1..1 + crcdata_len]);
        let i = 1 + crcdata_len;
        pkt[i] = (crc >> 24) as u8;
        pkt[i + 1] = (crc >> 16) as u8;
        pkt[i + 2] = (crc >> 8) as u8;
        pkt[i + 3] = crc as u8;
        if kind == SsdvPacketKind::WithFec {
            // Append RS parity.
            let mut block = [0u8; 255];
            block[..223].copy_from_slice(&pkt[1..1 + 223]);
            rs8::encode(&mut block).expect("rs encode");
            pkt[1 + 223..1 + 255].copy_from_slice(&block[223..255]);
        }
        pkt
    }

    #[test]
    fn callsign_round_trip() {
        let code = encode_callsign("HSCAT1");
        let decoded = decode_callsign(code);
        assert_eq!(decoded, "HSCAT1");
    }

    #[test]
    fn nofec_packet_decodes() {
        let pkt = build_packet(SsdvPacketKind::NoFec, 0x10);
        let decoded = SsdvDecoder::new().decode(&pkt).expect("decode");
        assert_eq!(decoded.kind, SsdvPacketKind::NoFec);
        assert_eq!(decoded.callsign, "HSCAT1");
        assert_eq!(decoded.image_id, 0x42);
        assert_eq!(decoded.packet_id, 5);
        assert_eq!(decoded.width, 320);
        assert_eq!(decoded.height, 240);
        assert_eq!(decoded.quality, 4);
        assert_eq!(decoded.mcu_mode, 0);
        assert_eq!(decoded.mcu_id, 5);
        assert!(decoded.crc_ok);
        assert_eq!(decoded.rs_errors, None);
        assert_eq!(decoded.payload.len(), 237);
    }

    #[test]
    fn fec_packet_decodes_clean() {
        let pkt = build_packet(SsdvPacketKind::WithFec, 0x20);
        let decoded = SsdvDecoder::new().decode(&pkt).expect("decode");
        assert_eq!(decoded.kind, SsdvPacketKind::WithFec);
        assert!(decoded.crc_ok);
        assert_eq!(decoded.rs_errors, Some(0));
        assert_eq!(decoded.payload.len(), 205);
    }

    #[test]
    fn fec_packet_corrects_errors() {
        let mut pkt = build_packet(SsdvPacketKind::WithFec, 0x33);
        let positions = [10usize, 50, 100, 150, 200];
        for &p in positions.iter() {
            pkt[p] ^= 0xA5;
        }
        let decoded = SsdvDecoder::new().decode(&pkt).expect("decode");
        assert_eq!(decoded.rs_errors, Some(positions.len() as u32));
        assert!(decoded.crc_ok);
    }

    #[test]
    fn rs_recovers_flipped_type_byte() {
        // Type byte flipped 0x66 -> 0x67 with no other corruption.
        // The tier-3 fallback forces pkt[1] = 0x66 before running
        // RS, so RS itself sees a clean codeword and reports zero
        // corrected errors. The packet still decodes successfully —
        // the previous logic would have rejected it as NoFec with a
        // bad CRC.
        let mut pkt = build_packet(SsdvPacketKind::WithFec, 0x55);
        pkt[1] = 0x67;
        let decoded = SsdvDecoder::new().decode(&pkt).expect("decode");
        assert_eq!(decoded.kind, SsdvPacketKind::WithFec);
        assert!(decoded.crc_ok);
        assert_eq!(decoded.rs_errors, Some(0));
    }

    #[test]
    fn rs_recovers_garbage_type_byte_plus_payload_errors() {
        // Type byte clobbered to a value that is neither 0x66 nor
        // 0x67, plus three real byte errors in the payload. Tier 3
        // resets pkt[1] for free, then RS corrects the three real
        // errors. fsphil silently fixes this case; we must match.
        let mut pkt = build_packet(SsdvPacketKind::WithFec, 0x77);
        pkt[1] = 0xAA;
        for i in 0..3 {
            pkt[40 + i * 30] ^= 0xA5;
        }
        let decoded = SsdvDecoder::new().decode(&pkt).expect("decode");
        assert_eq!(decoded.kind, SsdvPacketKind::WithFec);
        assert!(decoded.crc_ok);
        assert_eq!(decoded.rs_errors, Some(3));
    }

    #[test]
    fn rejects_uncorrectable() {
        // 17 byte errors exceeds RS(255,223)'s correction power.
        let mut pkt = build_packet(SsdvPacketKind::WithFec, 0);
        for i in 0..17 {
            pkt[10 + i * 5] ^= 0xA5;
        }
        assert_eq!(
            SsdvDecoder::new().decode(&pkt).unwrap_err(),
            SsdvDecodeError::Uncorrectable
        );
    }

    #[test]
    fn rejects_bad_length() {
        let buf = [0u8; 100];
        assert_eq!(
            SsdvDecoder::new().decode(&buf).unwrap_err(),
            SsdvDecodeError::BadLength
        );
    }
}
