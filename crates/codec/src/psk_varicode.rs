//! PSK31 / PSK63 Varicode decoder.
//!
//! PSK31 (G3PLX, 1998) is a low-baudrate DBPSK keyboard chat mode used
//! widely on HF and occasionally as a satellite beacon mode. PSK63 is
//! the same protocol at 2× baudrate. Both wrap text in a self-synchronising
//! Varicode: a variable-length prefix code where every codeword ends in
//! `"00"` and no internal pair of zeros appears, so a `"00"` run on the
//! wire is always a character boundary.
//!
//! This module is **codec layer only** — it consumes a stream of hard
//! bits already demodulated by [`openhoshimi_dsp::LinearDemodulator`]
//! configured for DBPSK at 31.25 or 62.5 baud, and produces ASCII text.
//! The bit convention follows G3PLX's original spec:
//!
//!   * `1` = no phase change between consecutive symbols.
//!   * `0` = 180° phase flip between consecutive symbols.
//!
//! The Varicode table below was copied from the G3PLX original
//! publication (CQ Magazine, 1998) and verified against fldigi's
//! `varicode.h`. It maps codewords (without the trailing `"00"`) to
//! ASCII codepoints 0..=127.
//!
//! Reference:
//! - <http://aintel.bi.ehu.es/psk31.html>
//! - fldigi varicode tables: `src/include/varicode.h`

use std::collections::HashMap;
use std::sync::OnceLock;

/// PSK31 nominal baudrate.
pub const PSK31_BAUDRATE: f32 = 31.25;
/// PSK63 nominal baudrate.
pub const PSK63_BAUDRATE: f32 = 62.5;

/// Streaming Varicode decoder.
///
/// Feed symbol bits from a DBPSK demodulator and pull out decoded text.
pub struct PskVaricodeDecoder {
    bit_buffer: String,
    output: String,
}

impl Default for PskVaricodeDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl PskVaricodeDecoder {
    /// Create an empty decoder.
    pub fn new() -> Self {
        Self {
            bit_buffer: String::new(),
            output: String::new(),
        }
    }

    /// Push hard symbol bits (`0` / `1`). Any newly decoded
    /// characters are returned as a fresh [`String`]; the cumulative
    /// text is also kept and can be retrieved with
    /// [`Self::full_text`].
    pub fn push_bits(&mut self, bits: &[u8]) -> String {
        let start = self.output.len();
        let table = decode_table();
        for &bit in bits {
            self.bit_buffer.push(if bit & 1 == 0 { '0' } else { '1' });
            // A character boundary is "00". On finding it, attempt
            // to decode everything up to the boundary; on failure,
            // drop the leading bit and keep scanning.
            while let Some(boundary) = self.bit_buffer.find("00") {
                let (codeword, remainder) = self.bit_buffer.split_at(boundary);
                if codeword.is_empty() {
                    // Leading "00" before any symbol: just skip the
                    // first zero and continue. This is the idle
                    // pattern PSK31 transmits between letters.
                    let new_buf = remainder[1..].to_string();
                    self.bit_buffer = new_buf;
                    continue;
                }
                if let Some(&ch) = table.get(codeword) {
                    self.output.push(ch);
                    // Drop the codeword and the first '0' of "00";
                    // keep the second '0' as the (possible) start
                    // of the next codeword's leading idle.
                    let keep = remainder[1..].to_string();
                    self.bit_buffer = keep;
                } else {
                    // Bad codeword: drop the first bit and rescan.
                    self.bit_buffer.remove(0);
                }
            }
            // Keep the buffer bounded so a long stretch of all-1s
            // (no character boundary in sight) does not grow without
            // limit. PSK31's longest standard codeword is 11 bits.
            if self.bit_buffer.len() > 64 {
                self.bit_buffer.drain(..self.bit_buffer.len() - 32);
            }
        }
        self.output[start..].to_string()
    }

    /// Cumulative decoded text.
    pub fn full_text(&self) -> &str {
        &self.output
    }
}

/// Encode an ASCII string to its raw Varicode bit stream (without the
/// trailing inter-character `"00"` separator). Mostly useful for
/// tests; transmitters insert idle `"0000..."` gaps as desired.
pub fn varicode_encode_bits(text: &str) -> Vec<u8> {
    let table = encode_table();
    let mut bits = Vec::new();
    for ch in text.chars() {
        let codeword = match table.get(&(ch as u32)) {
            Some(c) => *c,
            None => continue,
        };
        for b in codeword.bytes() {
            bits.push(if b == b'0' { 0u8 } else { 1u8 });
        }
        // Inter-character separator.
        bits.push(0);
        bits.push(0);
    }
    bits
}

fn decode_table() -> &'static HashMap<&'static str, char> {
    static TABLE: OnceLock<HashMap<&'static str, char>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut map = HashMap::new();
        for (idx, code) in VARICODE_TABLE.iter().enumerate() {
            if !code.is_empty() {
                map.insert(*code, idx as u8 as char);
            }
        }
        map
    })
}

fn encode_table() -> &'static HashMap<u32, &'static str> {
    static TABLE: OnceLock<HashMap<u32, &'static str>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut map = HashMap::new();
        for (idx, code) in VARICODE_TABLE.iter().enumerate() {
            if !code.is_empty() {
                map.insert(idx as u32, *code);
            }
        }
        map
    })
}

/// Varicode table indexed by ASCII codepoint. Empty entries are
/// non-printable codepoints PSK31 does not transmit.
///
/// Source: G3PLX original PSK31 specification (CQ Magazine 1998),
/// cross-checked against fldigi `src/include/varicode.h`.
const VARICODE_TABLE: &[&str] = &[
    "1010101011", // 0  NUL
    "1011011011", // 1  SOH
    "1011101101", // 2  STX
    "1101110111", // 3  ETX
    "1011101011", // 4  EOT
    "1101011111", // 5  ENQ
    "1011101111", // 6  ACK
    "1011111101", // 7  BEL
    "1011111111", // 8  BS
    "11101111",   // 9  HT
    "11101",      // 10 LF
    "1101101111", // 11 VT
    "1011011101", // 12 FF
    "11111",      // 13 CR
    "1101110101", // 14 SO
    "1110101011", // 15 SI
    "1011110111", // 16 DLE
    "1011110101", // 17 DC1
    "1110101101", // 18 DC2
    "1110101111", // 19 DC3
    "1101011011", // 20 DC4
    "1101101011", // 21 NAK
    "1101101101", // 22 SYN
    "1101010111", // 23 ETB
    "1101111011", // 24 CAN
    "1101111101", // 25 EM
    "1110110111", // 26 SUB
    "1101010101", // 27 ESC
    "1101011101", // 28 FS
    "1110111011", // 29 GS
    "1011111011", // 30 RS
    "1101111111", // 31 US
    "1",          // 32 SPACE
    "111111111",  // 33 !
    "101011111",  // 34 "
    "111110101",  // 35 #
    "111011011",  // 36 $
    "1011010101", // 37 %
    "1010111011", // 38 &
    "101111111",  // 39 '
    "11111011",   // 40 (
    "11110111",   // 41 )
    "101101111",  // 42 *
    "111011111",  // 43 +
    "1110101",    // 44 ,
    "110101",     // 45 -
    "1010111",    // 46 .
    "110101111",  // 47 /
    "10110111",   // 48 0
    "10111101",   // 49 1
    "11101101",   // 50 2
    "11111111",   // 51 3
    "101110111",  // 52 4
    "101011011",  // 53 5
    "101101011",  // 54 6
    "110101101",  // 55 7
    "110101011",  // 56 8
    "110110111",  // 57 9
    "11110101",   // 58 :
    "110111101",  // 59 ;
    "111101101",  // 60 <
    "1010101",    // 61 =
    "111010111",  // 62 >
    "1010101111", // 63 ?
    "1010111101", // 64 @
    "1111101",    // 65 A
    "11101011",   // 66 B
    "10101101",   // 67 C
    "10110101",   // 68 D
    "1110111",    // 69 E
    "11011011",   // 70 F
    "11111101",   // 71 G
    "101010101",  // 72 H
    "1111111",    // 73 I
    "111111101",  // 74 J
    "101111101",  // 75 K
    "11010111",   // 76 L
    "10111011",   // 77 M
    "11011101",   // 78 N
    "10101011",   // 79 O
    "11010101",   // 80 P
    "111011101",  // 81 Q
    "10101111",   // 82 R
    "1101111",    // 83 S
    "1101101",    // 84 T
    "101010111",  // 85 U
    "110110101",  // 86 V
    "101011101",  // 87 W
    "101110101",  // 88 X
    "101111011",  // 89 Y
    "1010101101", // 90 Z
    "111110111",  // 91 [
    "111101111",  // 92 \
    "111111011",  // 93 ]
    "1010111111", // 94 ^
    "101101101",  // 95 _
    "1011011111", // 96 `
    "1011",       // 97 a
    "1011111",    // 98 b
    "101111",     // 99 c
    "101101",     // 100 d
    "11",         // 101 e
    "111101",     // 102 f
    "1011011",    // 103 g
    "101011",     // 104 h
    "1101",       // 105 i
    "111101011",  // 106 j
    "10111111",   // 107 k
    "11011",      // 108 l
    "111011",     // 109 m
    "1111",       // 110 n
    "111",        // 111 o
    "111111",     // 112 p
    "110111111",  // 113 q
    "10101",      // 114 r
    "10111",      // 115 s
    "101",        // 116 t
    "110111",     // 117 u
    "1111011",    // 118 v
    "1101011",    // 119 w
    "11011111",   // 120 x
    "1011101",    // 121 y
    "111010101",  // 122 z
    "1010110111", // 123 {
    "110111011",  // 124 |
    "1010110101", // 125 }
    "1011010111", // 126 ~
    "1110110101", // 127 DEL
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varicode_table_lookup_roundtrip() {
        for (idx, code) in VARICODE_TABLE.iter().enumerate() {
            if code.is_empty() {
                continue;
            }
            let ch = idx as u8 as char;
            let bits = varicode_encode_bits(&ch.to_string());
            // The encoder appends a "00" gap; trim it for the
            // comparison since the table entry doesn't include it.
            assert!(bits.ends_with(&[0, 0]));
            let body: String = bits[..bits.len() - 2]
                .iter()
                .map(|b| if *b == 0 { '0' } else { '1' })
                .collect();
            assert_eq!(body, *code, "encode mismatch for {}", idx);
        }
    }

    #[test]
    fn round_trip_text() {
        let text = "Hello, OpenHoshimi! 73 de OH";
        let mut bits = vec![0u8; 16]; // leading idle
        bits.extend(varicode_encode_bits(text));
        bits.extend(vec![0u8; 16]); // trailing idle

        let mut decoder = PskVaricodeDecoder::new();
        decoder.push_bits(&bits);
        let decoded = decoder.full_text().to_string();
        assert_eq!(decoded, text, "decoded={decoded:?}");
    }

    #[test]
    fn handles_garbage_prefix() {
        // Junk before the message followed by an idle "00" gap.
        // After enough idle bits the decoder is at a clean character
        // boundary and the real text comes through verbatim.
        let text = "OK";
        let mut bits = vec![1u8, 0, 1, 0, 1, 1, 0, 1, 0]; // junk
        bits.extend(vec![0u8; 16]); // idle gap forces resync
        bits.extend(varicode_encode_bits(text));
        bits.extend(vec![0u8; 8]); // tail

        let mut decoder = PskVaricodeDecoder::new();
        decoder.push_bits(&bits);
        let decoded = decoder.full_text();
        assert!(
            decoded.contains(text),
            "expected to find {text:?} in decoded {decoded:?}"
        );
    }

    #[test]
    fn streaming_chunks_preserve_state() {
        let text = "DE PSK31 ";
        let mut bits = vec![0u8; 8];
        bits.extend(varicode_encode_bits(text));
        bits.extend(vec![0u8; 8]);
        let mut decoder = PskVaricodeDecoder::new();
        for chunk in bits.chunks(3) {
            decoder.push_bits(chunk);
        }
        assert_eq!(decoder.full_text(), text);
    }
}
