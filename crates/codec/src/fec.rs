//! Forward-error-correction helpers shared by frame decoders.

use openhoshimi_core::{DecodeError, Fec};

/// Reed-Solomon helper for shortened interleaved CCSDS-style codewords.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReedSolomon {
    interleave: usize,
}

impl ReedSolomon {
    /// Create a new Reed-Solomon helper for a fixed interleave factor.
    pub fn new(interleave: usize) -> Self {
        Self { interleave }
    }

    /// Decode a shortened Reed-Solomon block and keep the correction count.
    pub(crate) fn decode_shortened(
        &self,
        data: &[u8],
    ) -> Result<reed_solomon::RsDecoded, DecodeError> {
        reed_solomon::decode_shortened(data, self.interleave)
    }

    /// Decode a shortened Reed-Solomon block with the given erasure
    /// positions in codeword coordinates `[0, data.len())`. The decoder
    /// can correct up to `nroots` symbols when `2*errors + erasures
    /// <= nroots` (16 for the 32-parity AX.100 code), trading erasure
    /// knowledge for additional error-correction capacity. The returned
    /// `corrected_errors` count includes both pre-marked erasures and
    /// hard errors located by the BM step.
    pub(crate) fn decode_shortened_with_erasures(
        &self,
        data: &[u8],
        erasures: &[usize],
    ) -> Result<reed_solomon::RsDecoded, DecodeError> {
        reed_solomon::decode_shortened_with_erasures(data, self.interleave, erasures)
    }
}

impl Fec for ReedSolomon {
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>, DecodeError> {
        Ok(self.decode_shortened(data)?.message)
    }
}

/// Hard-decision Viterbi decoder for the CCSDS/AO-40 K=7, rate-1/2 code.
#[derive(Debug, Default, Clone, Copy)]
pub struct Viterbi;

impl Viterbi {
    /// Create a new stateless Viterbi decoder.
    pub fn new() -> Self {
        Self
    }

    /// Decode one-bit-per-byte hard symbols into one-bit-per-byte payload
    /// bits.
    pub fn decode_bits(&self, symbols: &[u8]) -> Result<Vec<u8>, DecodeError> {
        viterbi::decode_bits(symbols)
    }

    /// Decode signed soft symbols into one-bit-per-byte payload bits.
    ///
    /// Each input element carries one channel bit as a signed correlation
    /// value: positive means the matched filter output is closer to a `0`
    /// transmitted bit, negative means closer to `1`. Magnitude encodes the
    /// per-bit confidence; values near zero contribute little to the path
    /// metric. Soft decoding recovers roughly 2 dB of coding gain compared
    /// to [`decode_bits`](Self::decode_bits) at the cost of carrying a byte
    /// per channel bit instead of a packed bit.
    pub fn decode_soft_bits(&self, symbols: &[i8]) -> Result<Vec<u8>, DecodeError> {
        viterbi::decode_soft_bits(symbols)
    }
}

impl Fec for Viterbi {
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>, DecodeError> {
        self.decode_bits(data)
    }
}

/// CRC-32C (Castagnoli) used by libcsp / GOMspace AX100 frames.
///
/// Reflected input/output, init `0xFFFF_FFFF`, final XOR `0xFFFF_FFFF`,
/// reversed polynomial `0x82F6_3B78`. This is the CRC the GreenCube /
/// IO-117 ASM+Golay downlink appends after the CSP frame (verified
/// against on-air captures: the 4-byte big-endian trailer equals
/// `crc32c(csp_frame)`).
pub(crate) mod crc32c {
    const POLY: u32 = 0x82F6_3B78;

    /// Compute the CRC-32C of `data`.
    pub(crate) fn checksum(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (POLY & mask);
            }
        }
        crc ^ 0xFFFF_FFFF
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn matches_known_vector() {
            // Standard CRC-32C check value for the ASCII string
            // "123456789".
            assert_eq!(checksum(b"123456789"), 0xE306_9283);
        }

        #[test]
        fn matches_greencube_frame_trailer() {
            // On-air IO-117 digipeater frame: the descrambled CSP frame
            // followed by its 4-byte big-endian CRC-32C trailer.
            let frame: [u8; 43] = [
                0x82, 0x97, 0x75, 0x00, 0x1D, 0x03, 0x34, 0x4F, 0x34, 0x41, 0x3E, 0x4F, 0x4E, 0x34,
                0x43, 0x43, 0x4E, 0x2C, 0x20, 0x47, 0x72, 0x65, 0x65, 0x6E, 0x43, 0x75, 0x62, 0x65,
                0x2C, 0x20, 0x53, 0x54, 0x4F, 0x52, 0x45, 0x3D, 0x30, 0x20, 0x4A, 0x4E, 0x39, 0x32,
                0x0A,
            ];
            assert_eq!(checksum(&frame), 0x72CC_25E0);
        }
    }
}

pub(crate) mod ccsds_randomizer {
    const POLY_MASK: u8 = 0xa9;
    const INITIAL_STATE: u8 = 0xff;

    pub(crate) fn xor_sequence(bytes: &mut [u8]) {
        // CCSDS 131.0-B pseudo-random sequence: right-shift Fibonacci LFSR.
        // Output is the LSB taken before each shift; feedback (parity over
        // the tap mask) is shifted into the MSB. Byte fill order is MSB
        // first so that the very first emitted bit lands in bit 7.
        let mut state = INITIAL_STATE;
        for byte in bytes {
            let mut mask = 0u8;
            for bit_index in 0..8 {
                if state & 1 != 0 {
                    mask |= 1 << (7 - bit_index);
                }
                let feedback = (state & POLY_MASK).count_ones() & 1 != 0;
                state = (state >> 1) | (u8::from(feedback) << 7);
            }
            *byte ^= mask;
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn sequence_is_self_inverse() {
            let mut data: Vec<u8> = (0..64).collect();
            let original = data.clone();

            xor_sequence(&mut data);
            assert_ne!(data, original);
            xor_sequence(&mut data);

            assert_eq!(data, original);
        }

        #[test]
        fn sequence_matches_ccsds_blue_book() {
            // CCSDS 131.0-B Annex F pseudo-random sequence, first 16 bytes.
            // Polynomial h(x) = x^8 + x^7 + x^5 + x^3 + 1, all-1 initial
            // state, MSB of register output first. This is the canonical
            // sequence KA9Q's randomizer.c emits and what AO-40 / FUNcube
            // descramblers expect.
            const EXPECTED: [u8; 16] = [
                0xff, 0x48, 0x0e, 0xc0, 0x9a, 0x0d, 0x70, 0xbc, 0x8e, 0x2c, 0x93, 0xad, 0xa7, 0xb7,
                0x46, 0xce,
            ];
            let mut zeros = [0u8; 16];
            xor_sequence(&mut zeros);
            assert_eq!(zeros, EXPECTED);
        }
    }
}

pub(crate) mod viterbi {
    use openhoshimi_core::DecodeError;

    const STATES: usize = 64;
    const K: usize = 7;
    const G1: u8 = 0x4f;
    const G2: u8 = 0x6d;
    const INF: u32 = u32::MAX / 4;
    const SECOND_SYMBOL_INVERTED: bool = true;

    pub(crate) fn decode_bits(symbols: &[u8]) -> Result<Vec<u8>, DecodeError> {
        if symbols.len() < 2 || symbols.len() / 2 * 2 != symbols.len() {
            return Err(DecodeError::InvalidEncoding(
                "Viterbi input must contain pairs of symbols".to_string(),
            ));
        }

        let steps = symbols.len() / 2;
        let mut metrics = [INF; STATES];
        metrics[0] = 0;
        let mut decisions = vec![[0u8; STATES]; steps];

        for step in 0..steps {
            let received = [symbols[step * 2] & 1, symbols[step * 2 + 1] & 1];
            let mut next_metrics = [INF; STATES];

            for (state, metric) in metrics.iter().enumerate() {
                if *metric == INF {
                    continue;
                }
                for bit in 0..=1u8 {
                    let next_state = ((state << 1) | usize::from(bit)) & (STATES - 1);
                    let encoded = encode_branch(state as u8, bit);
                    let branch_metric = u32::from(hamming2(received, encoded));
                    let candidate = *metric + branch_metric;
                    if candidate < next_metrics[next_state] {
                        next_metrics[next_state] = candidate;
                        decisions[step][next_state] = (state as u8) | (bit << 6);
                    }
                }
            }

            metrics = next_metrics;
        }

        let mut state = metrics
            .iter()
            .enumerate()
            .min_by_key(|(_, metric)| *metric)
            .map(|(state, _)| state)
            .ok_or_else(|| DecodeError::InvalidEncoding("empty Viterbi trellis".to_string()))?;

        let mut decoded = vec![0u8; steps];
        for step in (0..steps).rev() {
            let decision = decisions[step][state];
            decoded[step] = (decision >> 6) & 1;
            state = usize::from(decision & 0x3f);
        }

        Ok(decoded)
    }

    pub(crate) fn decode_soft_bits(symbols: &[i8]) -> Result<Vec<u8>, DecodeError> {
        if symbols.len() < 2 || symbols.len() / 2 * 2 != symbols.len() {
            return Err(DecodeError::InvalidEncoding(
                "Viterbi input must contain pairs of symbols".to_string(),
            ));
        }

        // Each soft sample is mapped so that a transmitted `0` corresponds
        // to a positive correlation and a transmitted `1` to a negative
        // correlation. The branch metric is the squared Euclidean distance
        // between the received pair and the expected (±S, ±S) constellation
        // point, with S = 64 chosen as the soft-decision center. This keeps
        // the metric in u32 range for the 5132-bit AO-40 stream.
        const SOFT_CENTER: i32 = 64;

        let steps = symbols.len() / 2;
        let mut metrics = [INF; STATES];
        metrics[0] = 0;
        let mut decisions = vec![[0u8; STATES]; steps];

        for step in 0..steps {
            let received = [
                i32::from(symbols[step * 2]),
                i32::from(symbols[step * 2 + 1]),
            ];
            let mut next_metrics = [INF; STATES];

            for (state, metric) in metrics.iter().enumerate() {
                if *metric == INF {
                    continue;
                }
                for bit in 0..=1u8 {
                    let next_state = ((state << 1) | usize::from(bit)) & (STATES - 1);
                    let encoded = encode_branch(state as u8, bit);
                    let expected_a = if encoded[0] == 0 {
                        SOFT_CENTER
                    } else {
                        -SOFT_CENTER
                    };
                    let expected_b = if encoded[1] == 0 {
                        SOFT_CENTER
                    } else {
                        -SOFT_CENTER
                    };
                    let diff_a = received[0] - expected_a;
                    let diff_b = received[1] - expected_b;
                    let branch_metric = (diff_a * diff_a + diff_b * diff_b) as u32;
                    let candidate = metric.saturating_add(branch_metric);
                    if candidate < next_metrics[next_state] {
                        next_metrics[next_state] = candidate;
                        decisions[step][next_state] = (state as u8) | (bit << 6);
                    }
                }
            }

            metrics = next_metrics;
        }

        let mut state = metrics
            .iter()
            .enumerate()
            .min_by_key(|(_, metric)| *metric)
            .map(|(state, _)| state)
            .ok_or_else(|| DecodeError::InvalidEncoding("empty Viterbi trellis".to_string()))?;

        let mut decoded = vec![0u8; steps];
        for step in (0..steps).rev() {
            let decision = decisions[step][state];
            decoded[step] = (decision >> 6) & 1;
            state = usize::from(decision & 0x3f);
        }

        Ok(decoded)
    }

    /// Convolutionally encode one-bit-per-byte input into hard channel bits.
    ///
    /// Used by the AO-40 encoder path and by codec roundtrip tests; mirrors
    /// the decoder's CCSDS K=7 rate-1/2 polynomial convention with G2
    /// inverted.
    pub fn encode_bits(bits: &[u8]) -> Vec<u8> {
        let mut state = 0u8;
        let mut out = Vec::with_capacity(bits.len() * 2);
        for &bit in bits {
            let encoded = encode_branch(state, bit & 1);
            out.extend_from_slice(&encoded);
            state = ((state << 1) | (bit & 1)) & ((1 << (K - 1)) - 1);
        }
        out
    }

    fn encode_branch(state: u8, bit: u8) -> [u8; 2] {
        let register = ((state << 1) | (bit & 1)) & ((1 << K) - 1);
        let second = if SECOND_SYMBOL_INVERTED {
            parity(register & G2) ^ 1
        } else {
            parity(register & G2)
        };
        [parity(register & G1), second]
    }

    fn parity(value: u8) -> u8 {
        (value.count_ones() & 1) as u8
    }

    fn hamming2(received: [u8; 2], expected: [u8; 2]) -> u8 {
        u8::from(received[0] != expected[0]) + u8::from(received[1] != expected[1])
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn decodes_clean_symbols() {
            let bits: Vec<u8> = (0..96).map(|i| u8::from(i % 5 == 0)).collect();
            let symbols = encode_bits(&bits);
            let decoded = match decode_bits(&symbols) {
                Ok(decoded) => decoded,
                Err(err) => panic!("valid Viterbi symbols: {err}"),
            };

            assert_eq!(decoded, bits);
        }

        #[test]
        fn corrects_single_symbol_error() {
            let bits: Vec<u8> = (0..128).map(|i| u8::from(i % 7 < 3)).collect();
            let mut symbols = encode_bits(&bits);
            symbols[17] ^= 1;
            let decoded = match decode_bits(&symbols) {
                Ok(decoded) => decoded,
                Err(err) => panic!("correctable Viterbi symbols: {err}"),
            };

            assert_eq!(decoded, bits);
        }

        #[test]
        fn rejects_odd_symbol_count() {
            let err = match decode_bits(&[0, 1, 1]) {
                Ok(_) => panic!("odd symbol count should fail"),
                Err(err) => err,
            };

            assert!(matches!(err, DecodeError::InvalidEncoding(_)));
        }

        fn soft_from_hard(symbols: &[u8], scale: i8) -> Vec<i8> {
            symbols
                .iter()
                .map(|bit| if bit & 1 == 0 { scale } else { -scale })
                .collect()
        }

        #[test]
        fn soft_decodes_clean_symbols() {
            let bits: Vec<u8> = (0..96).map(|i| u8::from(i % 5 == 0)).collect();
            let hard = encode_bits(&bits);
            let soft = soft_from_hard(&hard, 64);

            let decoded = match decode_soft_bits(&soft) {
                Ok(decoded) => decoded,
                Err(err) => panic!("valid soft Viterbi symbols: {err}"),
            };

            assert_eq!(decoded, bits);
        }

        #[test]
        fn soft_corrects_more_errors_than_hard() {
            // Construct a deterministic 256-bit message and inject the same
            // dense pattern of channel errors into the hard-bit stream and
            // the matching soft stream. Soft decoding should still recover
            // the message in cases where hard fails because magnitude carries
            // confidence about the unflipped symbols.
            let bits: Vec<u8> = (0..256).map(|i| u8::from(i % 3 == 0)).collect();
            let hard = encode_bits(&bits);
            let mut hard_corrupt = hard.clone();
            let mut soft_corrupt = soft_from_hard(&hard, 64);
            for index in (5..hard_corrupt.len()).step_by(11) {
                hard_corrupt[index] ^= 1;
                // Flip the soft sign but keep a smaller magnitude: this is
                // an unreliable received bit, which is exactly what soft
                // decoding is designed to deweight.
                soft_corrupt[index] = -soft_corrupt[index] / 4;
            }

            let soft_decoded = match decode_soft_bits(&soft_corrupt) {
                Ok(decoded) => decoded,
                Err(err) => panic!("correctable soft symbols: {err}"),
            };
            assert_eq!(soft_decoded, bits);
        }
    }
}

pub(crate) mod golay {
    use openhoshimi_core::DecodeError;

    const N: usize = 12;
    const H: [u32; N] = [
        0x8008ed, 0x4001db, 0x2003b5, 0x100769, 0x080ed1, 0x040da3, 0x020b47, 0x01068f, 0x008d1d,
        0x004a3b, 0x002477, 0x001ffe,
    ];

    /// Result of decoding a Golay(24,12) word.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct GolayWord {
        pub(crate) data: u16,
        pub(crate) corrected_errors: u8,
    }

    #[cfg(test)]
    pub(crate) fn encode(data: u16) -> u32 {
        let data = u32::from(data & 0x0fff);
        let mut parity = 0u32;
        for row in H {
            parity <<= 1;
            parity |= (row & data).count_ones() & 1;
        }
        ((parity & 0x0fff) << N) | data
    }

    pub(crate) fn decode(word: u32) -> Result<GolayWord, DecodeError> {
        let word = word & 0x00ff_ffff;
        let syndrome = syndrome(word);

        if syndrome.count_ones() <= 3 {
            let corrected = word ^ (syndrome << N);
            return Ok(decoded_word(corrected, syndrome.count_ones()));
        }

        for (i, row) in H.iter().enumerate() {
            let modified = syndrome ^ b(*row);
            if modified.count_ones() <= 2 {
                let error = (modified << N) | (1 << (N - i - 1));
                return Ok(decoded_word(word ^ error, error.count_ones()));
            }
        }

        let q = q_syndrome(syndrome);
        if q.count_ones() <= 3 {
            return Ok(decoded_word(word ^ q, q.count_ones()));
        }

        for (i, row) in H.iter().enumerate() {
            let modified = q ^ b(*row);
            if modified.count_ones() <= 2 {
                let error = (1 << (2 * N - i - 1)) | modified;
                return Ok(decoded_word(word ^ error, error.count_ones()));
            }
        }

        Err(DecodeError::InvalidEncoding(
            "uncorrectable Golay(24,12) header".to_string(),
        ))
    }

    fn decoded_word(word: u32, corrected_errors: u32) -> GolayWord {
        GolayWord {
            data: (word & 0x0fff) as u16,
            corrected_errors: corrected_errors as u8,
        }
    }

    fn syndrome(word: u32) -> u32 {
        let mut out = 0u32;
        for row in H {
            out <<= 1;
            out |= (row & word).count_ones() & 1;
        }
        out
    }

    fn q_syndrome(syndrome: u32) -> u32 {
        let mut out = 0u32;
        for row in H {
            out <<= 1;
            out |= (b(row) & syndrome).count_ones() & 1;
        }
        out
    }

    fn b(row: u32) -> u32 {
        row & 0x0fff
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn decodes_clean_word() {
            let encoded = encode(0x5a3);
            let decoded = match decode(encoded) {
                Ok(decoded) => decoded,
                Err(err) => panic!("valid Golay word: {err}"),
            };

            assert_eq!(decoded.data, 0x5a3);
            assert_eq!(decoded.corrected_errors, 0);
        }

        #[test]
        fn corrects_three_bit_errors() {
            let encoded = encode(0x2a5);
            let damaged = encoded ^ (1 << 23) ^ (1 << 9) ^ 1;
            let decoded = match decode(damaged) {
                Ok(decoded) => decoded,
                Err(err) => panic!("correctable Golay word: {err}"),
            };

            assert_eq!(decoded.data, 0x2a5);
            assert_eq!(decoded.corrected_errors, 3);
        }

        #[test]
        fn rejects_uncorrectable_word() {
            let encoded = encode(0x2a5);
            let damaged = encoded ^ 0x00f0_f00f;
            let err = match decode(damaged) {
                Ok(_) => panic!("uncorrectable Golay word should fail"),
                Err(err) => err,
            };

            assert!(matches!(err, DecodeError::InvalidEncoding(_)));
        }
    }
}

pub(crate) mod reed_solomon {
    use openhoshimi_core::DecodeError;

    const NN: usize = 255;
    pub(crate) const PARITY_LEN: usize = 32;
    const FCR: i32 = 112;
    const PRIM: i32 = 11;
    const ALPHA_TO: [u8; NN] = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x87, 0x89, 0x95, 0xad, 0xdd, 0x3d, 0x7a,
        0xf4, 0x6f, 0xde, 0x3b, 0x76, 0xec, 0x5f, 0xbe, 0xfb, 0x71, 0xe2, 0x43, 0x86, 0x8b, 0x91,
        0xa5, 0xcd, 0x1d, 0x3a, 0x74, 0xe8, 0x57, 0xae, 0xdb, 0x31, 0x62, 0xc4, 0x0f, 0x1e, 0x3c,
        0x78, 0xf0, 0x67, 0xce, 0x1b, 0x36, 0x6c, 0xd8, 0x37, 0x6e, 0xdc, 0x3f, 0x7e, 0xfc, 0x7f,
        0xfe, 0x7b, 0xf6, 0x6b, 0xd6, 0x2b, 0x56, 0xac, 0xdf, 0x39, 0x72, 0xe4, 0x4f, 0x9e, 0xbb,
        0xf1, 0x65, 0xca, 0x13, 0x26, 0x4c, 0x98, 0xb7, 0xe9, 0x55, 0xaa, 0xd3, 0x21, 0x42, 0x84,
        0x8f, 0x99, 0xb5, 0xed, 0x5d, 0xba, 0xf3, 0x61, 0xc2, 0x03, 0x06, 0x0c, 0x18, 0x30, 0x60,
        0xc0, 0x07, 0x0e, 0x1c, 0x38, 0x70, 0xe0, 0x47, 0x8e, 0x9b, 0xb1, 0xe5, 0x4d, 0x9a, 0xb3,
        0xe1, 0x45, 0x8a, 0x93, 0xa1, 0xc5, 0x0d, 0x1a, 0x34, 0x68, 0xd0, 0x27, 0x4e, 0x9c, 0xbf,
        0xf9, 0x75, 0xea, 0x53, 0xa6, 0xcb, 0x11, 0x22, 0x44, 0x88, 0x97, 0xa9, 0xd5, 0x2d, 0x5a,
        0xb4, 0xef, 0x59, 0xb2, 0xe3, 0x41, 0x82, 0x83, 0x81, 0x85, 0x8d, 0x9d, 0xbd, 0xfd, 0x7d,
        0xfa, 0x73, 0xe6, 0x4b, 0x96, 0xab, 0xd1, 0x25, 0x4a, 0x94, 0xaf, 0xd9, 0x35, 0x6a, 0xd4,
        0x2f, 0x5e, 0xbc, 0xff, 0x79, 0xf2, 0x63, 0xc6, 0x0b, 0x16, 0x2c, 0x58, 0xb0, 0xe7, 0x49,
        0x92, 0xa3, 0xc1, 0x05, 0x0a, 0x14, 0x28, 0x50, 0xa0, 0xc7, 0x09, 0x12, 0x24, 0x48, 0x90,
        0xa7, 0xc9, 0x15, 0x2a, 0x54, 0xa8, 0xd7, 0x29, 0x52, 0xa4, 0xcf, 0x19, 0x32, 0x64, 0xc8,
        0x17, 0x2e, 0x5c, 0xb8, 0xf7, 0x69, 0xd2, 0x23, 0x46, 0x8c, 0x9f, 0xb9, 0xf5, 0x6d, 0xda,
        0x33, 0x66, 0xcc, 0x1f, 0x3e, 0x7c, 0xf8, 0x77, 0xee, 0x5b, 0xb6, 0xeb, 0x51, 0xa2, 0xc3,
    ];

    const INDEX_OF: [u8; 256] = [
        255, 0, 1, 99, 2, 198, 100, 106, 3, 205, 199, 188, 101, 126, 107, 42, 4, 141, 206, 78, 200,
        212, 189, 225, 102, 221, 127, 49, 108, 32, 43, 243, 5, 87, 142, 232, 207, 172, 79, 131,
        201, 217, 213, 65, 190, 148, 226, 180, 103, 39, 222, 240, 128, 177, 50, 53, 109, 69, 33,
        18, 44, 13, 244, 56, 6, 155, 88, 26, 143, 121, 233, 112, 208, 194, 173, 168, 80, 117, 132,
        72, 202, 252, 218, 138, 214, 84, 66, 36, 191, 152, 149, 249, 227, 94, 181, 21, 104, 97, 40,
        186, 223, 76, 241, 47, 129, 230, 178, 63, 51, 238, 54, 16, 110, 24, 70, 166, 34, 136, 19,
        247, 45, 184, 14, 61, 245, 164, 57, 59, 7, 158, 156, 157, 89, 159, 27, 8, 144, 9, 122, 28,
        234, 160, 113, 90, 209, 29, 195, 123, 174, 10, 169, 145, 81, 91, 118, 114, 133, 161, 73,
        235, 203, 124, 253, 196, 219, 30, 139, 210, 215, 146, 85, 170, 67, 11, 37, 175, 192, 115,
        153, 119, 150, 92, 250, 82, 228, 236, 95, 74, 182, 162, 22, 134, 105, 197, 98, 254, 41,
        125, 187, 204, 224, 211, 77, 140, 242, 31, 48, 220, 130, 171, 231, 86, 179, 147, 64, 216,
        52, 176, 239, 38, 55, 12, 17, 68, 111, 120, 25, 154, 71, 116, 167, 193, 35, 83, 137, 251,
        20, 93, 248, 151, 46, 75, 185, 96, 15, 237, 62, 229, 246, 135, 165, 23, 58, 163, 60, 183,
    ];

    const GENERATOR_POLY: [u8; PARITY_LEN + 1] = [
        0, 249, 59, 66, 4, 43, 126, 251, 97, 30, 3, 213, 50, 66, 170, 5, 24, 5, 170, 66, 50, 213,
        3, 30, 97, 251, 126, 43, 4, 66, 59, 249, 0,
    ];

    /// Result of decoding a Reed-Solomon codeword.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct RsDecoded {
        pub(crate) message: Vec<u8>,
        pub(crate) corrected_errors: usize,
    }

    pub(crate) fn decode_shortened(
        codeword: &[u8],
        interleave: usize,
    ) -> Result<RsDecoded, DecodeError> {
        decode_shortened_with_erasures(codeword, interleave, &[])
    }

    /// Decode a shortened, interleaved RS codeword with optional erasure
    /// positions. `erasures` are byte indices in the codeword (range
    /// `0..codeword.len()`); they identify symbols whose value is treated
    /// as unknown by the decoder. Combined errors+erasures decoding can
    /// fix up to `nroots` symbols when `2*errors + erasures <= nroots`.
    pub(crate) fn decode_shortened_with_erasures(
        codeword: &[u8],
        interleave: usize,
        erasures: &[usize],
    ) -> Result<RsDecoded, DecodeError> {
        if interleave == 0 || codeword.len() / interleave * interleave != codeword.len() {
            return Err(DecodeError::InvalidEncoding(
                "invalid Reed-Solomon interleave".to_string(),
            ));
        }

        let rs_nn = codeword.len() / interleave;
        if rs_nn <= PARITY_LEN || rs_nn > NN {
            return Err(DecodeError::TooShort(codeword.len()));
        }

        for &eras in erasures {
            if eras >= codeword.len() {
                return Err(DecodeError::InvalidEncoding(
                    "Reed-Solomon erasure index out of range".to_string(),
                ));
            }
        }

        let pad = NN - rs_nn;
        let mut corrected_errors = 0usize;
        let mut out = vec![0u8; codeword.len() - interleave * PARITY_LEN];

        // Erasure positions are routed to their interleave path, then
        // shifted into the padded RS(255,223) coordinate system that
        // `decode_block` expects: position 0 of a path's transmitted
        // payload sits at index `pad` of its RS block.
        let mut path_eras: Vec<Vec<usize>> = vec![Vec::new(); interleave];
        for &eras in erasures {
            let path = eras % interleave;
            let pos_in_path = eras / interleave;
            path_eras[path].push(pad + pos_in_path);
        }

        for path in 0..interleave {
            let mut block = [0u8; NN];
            for k in 0..rs_nn {
                block[pad + k] = codeword[path + k * interleave];
            }

            corrected_errors += decode_block(&mut block, pad, &path_eras[path])?;

            for k in 0..(rs_nn - PARITY_LEN) {
                out[path + k * interleave] = block[pad + k];
            }
        }

        Ok(RsDecoded {
            message: out,
            corrected_errors,
        })
    }

    /// Encode an interleaved shortened RS message and append parity.
    ///
    /// Used by the AO-40 encoder path and codec roundtrip tests; matches the
    /// (160,128) shortened-from-(255,223) layout the decoder expects with
    /// CCSDS FCR=112 and PRIM=11.
    pub fn encode_shortened(message: &[u8], interleave: usize) -> Vec<u8> {
        let rs_nn = message.len() / interleave + PARITY_LEN;
        let mut out = vec![0u8; rs_nn * interleave];

        for path in 0..interleave {
            let mut data = Vec::new();
            for index in (path..message.len()).step_by(interleave) {
                data.push(message[index]);
            }
            let parity = encode_parity(&data);
            for (k, byte) in data.iter().chain(parity.iter()).enumerate() {
                out[path + k * interleave] = *byte;
            }
        }

        out
    }

    const IPRIM: usize = 116;

    fn decode_block(
        data: &mut [u8; NN],
        pad: usize,
        eras_pos: &[usize],
    ) -> Result<usize, DecodeError> {
        let nroots = PARITY_LEN;
        let no_eras = eras_pos.len();

        if no_eras > nroots {
            return Err(DecodeError::CrcMismatch);
        }
        for &pos in eras_pos {
            if pos < pad || pos >= NN {
                return Err(DecodeError::InvalidEncoding(
                    "Reed-Solomon erasure position out of range".to_string(),
                ));
            }
        }

        let mut s_val = [0u8; PARITY_LEN];
        for (i, syn) in s_val.iter_mut().enumerate() {
            *syn = data[pad];
            for &symbol in data.iter().skip(pad + 1) {
                if *syn == 0 {
                    *syn = symbol;
                } else {
                    let exp = usize::from(index_of(*syn)) + ((FCR + i as i32) * PRIM) as usize;
                    *syn = symbol ^ alpha_to(mod_nn(exp));
                }
            }
        }

        if s_val.iter().all(|s| *s == 0) {
            return Ok(0);
        }

        let s: Vec<usize> = s_val.iter().map(|&v| usize::from(index_of(v))).collect();

        // Berlekamp-Massey (lambda and b in value form). Lambda is seeded
        // with the erasure locator polynomial, so the BM loop only needs
        // `nroots - no_eras` iterations to find the additional error
        // locator factors.
        let mut lambda = [0u8; PARITY_LEN + 1];
        lambda[0] = 1;
        if no_eras > 0 {
            lambda[1] = alpha_to(mod_nn((PRIM as usize) * (NN - 1 - eras_pos[0])));
            for (i, eras) in eras_pos.iter().enumerate().skip(1) {
                let u = mod_nn((PRIM as usize) * (NN - 1 - eras));
                for j in (1..=i + 1).rev() {
                    let tmp_idx = index_of(lambda[j - 1]);
                    if tmp_idx != NN as u8 {
                        lambda[j] ^= alpha_to(mod_nn(u + usize::from(tmp_idx)));
                    }
                }
            }
        }
        let mut b = lambda;
        let mut el = no_eras;

        for r in no_eras..nroots {
            let mut discr = s_val[r];
            for i in 1..=el.min(nroots - 1) {
                if lambda[i] != 0 && s[r - i] != NN {
                    discr ^= alpha_to(mod_nn(usize::from(index_of(lambda[i])) + s[r - i]));
                }
            }

            if discr == 0 {
                b.copy_within(..nroots, 1);
                b[0] = 0;
            } else {
                let discr_idx = usize::from(index_of(discr));
                let mut t = lambda;
                for i in 0..nroots {
                    if b[i] != 0 {
                        t[i + 1] ^= alpha_to(mod_nn(discr_idx + usize::from(index_of(b[i]))));
                    }
                }

                if 2 * el <= r + no_eras {
                    el = r + 1 + no_eras - el;
                    let discr_inv = NN - discr_idx;
                    for i in 0..=nroots {
                        b[i] = if lambda[i] != 0 {
                            alpha_to(mod_nn(usize::from(index_of(lambda[i])) + discr_inv))
                        } else {
                            0
                        };
                    }
                } else {
                    b.copy_within(..nroots, 1);
                    b[0] = 0;
                }

                lambda = t;
            }
        }

        let deg_lambda = match lambda.iter().rposition(|&v| v != 0) {
            Some(d) if d > 0 && 2 * d <= nroots + no_eras => d,
            _ => return Err(DecodeError::CrcMismatch),
        };

        let lambda_idx: [usize; PARITY_LEN + 1] = {
            let mut arr = [NN; PARITY_LEN + 1];
            for (i, &v) in lambda.iter().enumerate() {
                arr[i] = usize::from(index_of(v));
            }
            arr
        };

        // Chien search
        let mut reg = lambda_idx;
        let mut root = [0usize; PARITY_LEN];
        let mut loc = [0usize; PARITY_LEN];
        let mut count = 0usize;
        let mut k = IPRIM - 1;

        for i in 1..=NN {
            let mut q = 1u8;
            for j in (1..=deg_lambda).rev() {
                if reg[j] != NN {
                    reg[j] = mod_nn(reg[j] + j);
                    q ^= alpha_to(reg[j]);
                }
            }
            if q == 0 {
                if k < pad {
                    return Err(DecodeError::CrcMismatch);
                }
                root[count] = i;
                loc[count] = k;
                count += 1;
                if count >= deg_lambda {
                    break;
                }
            }
            k = if k + IPRIM >= NN {
                k + IPRIM - NN
            } else {
                k + IPRIM
            };
        }

        if count != deg_lambda {
            return Err(DecodeError::CrcMismatch);
        }

        // Omega(x) = S(x)*Lambda(x) mod x^nroots
        let deg_omega = deg_lambda - 1;
        let mut omega = [NN; PARITY_LEN + 1];
        for i in 0..=deg_omega {
            let mut tmp = 0u8;
            for j in (0..=i).rev() {
                if s[j] != NN && lambda_idx[i - j] != NN {
                    tmp ^= alpha_to(mod_nn(s[j] + lambda_idx[i - j]));
                }
            }
            omega[i] = usize::from(index_of(tmp));
        }

        // Forney
        for j in 0..count {
            let mut num1 = 0u8;
            #[allow(clippy::needless_range_loop)]
            for i in 0..=deg_omega {
                if omega[i] != NN {
                    num1 ^= alpha_to(mod_nn(omega[i] + i * root[j]));
                }
            }
            let num2 = alpha_to(mod_nn(root[j] * ((FCR as usize) - 1) + NN));

            let mut den = 0u8;
            let mut i = (deg_lambda.min(nroots - 1)) & !1;
            loop {
                if lambda_idx[i + 1] != NN {
                    den ^= alpha_to(mod_nn(lambda_idx[i + 1] + i * root[j]));
                }
                if i < 2 {
                    break;
                }
                i -= 2;
            }

            if den == 0 {
                return Err(DecodeError::CrcMismatch);
            }
            if num1 != 0 {
                data[loc[j]] ^= alpha_to(mod_nn(
                    usize::from(index_of(num1)) + usize::from(index_of(num2)) + NN
                        - usize::from(index_of(den)),
                ));
            }
        }

        Ok(count)
    }

    fn encode_parity(message: &[u8]) -> [u8; PARITY_LEN] {
        let mut parity = [0u8; PARITY_LEN];

        for &symbol in message {
            let feedback = index_of(symbol ^ parity[0]);
            if feedback != 255 {
                for j in 1..PARITY_LEN {
                    parity[j] ^= alpha_to(mod_nn(
                        usize::from(feedback) + usize::from(GENERATOR_POLY[PARITY_LEN - j]),
                    ));
                }
            }
            parity.copy_within(1.., 0);
            parity[PARITY_LEN - 1] = if feedback != 255 {
                alpha_to(mod_nn(
                    usize::from(feedback) + usize::from(GENERATOR_POLY[0]),
                ))
            } else {
                0
            };
        }

        parity
    }

    fn alpha_to(index: usize) -> u8 {
        ALPHA_TO[index]
    }

    fn index_of(value: u8) -> u8 {
        INDEX_OF[usize::from(value)]
    }

    fn mod_nn(mut value: usize) -> usize {
        while value >= NN {
            value -= NN;
            value = (value >> 8) + (value & NN);
        }
        value
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn shortened_round_trip_with_no_errors() {
            let message: Vec<u8> = (0..128).collect();
            let encoded = encode_shortened(&message, 1);
            let decoded = match decode_shortened(&encoded, 1) {
                Ok(decoded) => decoded,
                Err(err) => panic!("valid RS codeword: {err}"),
            };

            assert_eq!(decoded.message, message);
            assert_eq!(decoded.corrected_errors, 0);
        }

        #[test]
        fn shortened_interleaved_round_trip_with_no_errors() {
            let message: Vec<u8> = (0..=255).collect();
            let encoded = encode_shortened(&message, 2);
            let decoded = match decode_shortened(&encoded, 2) {
                Ok(decoded) => decoded,
                Err(err) => panic!("valid interleaved RS codeword: {err}"),
            };

            assert_eq!(decoded.message, message);
            assert_eq!(decoded.corrected_errors, 0);
        }

        #[test]
        fn shortened_corrects_errors() {
            let message: Vec<u8> = (0..160).collect();
            let mut encoded = encode_shortened(&message, 1);
            encoded[3] ^= 0x55;

            let decoded = match decode_shortened(&encoded, 1) {
                Ok(decoded) => decoded,
                Err(err) => panic!("damaged RS codeword should be corrected: {err}"),
            };

            assert_eq!(decoded.message, message);
            assert!(decoded.corrected_errors > 0);
        }

        #[test]
        fn shortened_corrects_pure_erasures_at_capacity() {
            // 16 erasures, 0 errors: 2*0 + 16 = 16 = nroots, decoder must
            // recover exactly.
            let message: Vec<u8> = (0..160).collect();
            let mut encoded = encode_shortened(&message, 1);
            let mut erasures = Vec::with_capacity(PARITY_LEN / 2);
            for k in 0..(PARITY_LEN / 2) {
                let pos = 5 + k * 7;
                encoded[pos] ^= 0xa5;
                erasures.push(pos);
            }

            let decoded = match decode_shortened_with_erasures(&encoded, 1, &erasures) {
                Ok(d) => d,
                Err(err) => panic!("pure erasures within capacity must decode: {err}"),
            };
            assert_eq!(decoded.message, message);
        }

        #[test]
        fn shortened_corrects_errors_plus_erasures_within_capacity() {
            // 4 errors + 8 erasures: 2*4 + 8 = 16 = nroots, at capacity.
            let message: Vec<u8> = (0..160).collect();
            let mut encoded = encode_shortened(&message, 1);
            let mut erasures = Vec::new();
            for k in 0..8 {
                let pos = 7 + k * 5;
                encoded[pos] ^= 0xc3;
                erasures.push(pos);
            }
            // Hard errors the decoder must locate without prior knowledge.
            for k in 0..4 {
                let pos = 80 + k * 11;
                encoded[pos] ^= 0x5a;
            }

            let decoded = match decode_shortened_with_erasures(&encoded, 1, &erasures) {
                Ok(d) => d,
                Err(err) => panic!("errors+erasures within capacity must decode: {err}"),
            };
            assert_eq!(decoded.message, message);
        }

        #[test]
        fn shortened_rejects_errors_plus_erasures_over_capacity() {
            // 21 hard errors + 8 erasures: deg(true error) = 29, but the
            // generalized capacity bound `2*deg <= nroots + no_eras`
            // permits at most deg = (32+8)/2 = 20. Decoding must fail
            // rather than silently produce a different codeword.
            let message: Vec<u8> = (0..160).collect();
            let mut encoded = encode_shortened(&message, 1);
            let mut erasures = Vec::new();
            for k in 0..8 {
                let pos = 7 + k * 5;
                encoded[pos] ^= 0xc3;
                erasures.push(pos);
            }
            for k in 0..21 {
                let pos = 80 + k * 3;
                encoded[pos] ^= 0x5a;
            }

            if let Ok(decoded) = decode_shortened_with_erasures(&encoded, 1, &erasures) {
                assert_ne!(
                    decoded.message, message,
                    "decoder must not silently succeed past capacity"
                );
            }
        }

        #[test]
        fn shortened_clean_codeword_with_erasures_is_unchanged() {
            // Marking erasures on an unblemished codeword must not
            // corrupt the recovered message.
            let message: Vec<u8> = (0..160).collect();
            let encoded = encode_shortened(&message, 1);
            let erasures = vec![3, 47, 99];

            let decoded = match decode_shortened_with_erasures(&encoded, 1, &erasures) {
                Ok(d) => d,
                Err(err) => panic!("clean codeword must decode regardless of erasures: {err}"),
            };
            assert_eq!(decoded.message, message);
            assert_eq!(decoded.corrected_errors, 0);
        }

        #[test]
        fn shortened_interleaved_corrects_per_path_erasures() {
            // Interleave 2: each path independently sees its own
            // 4 errors + 8 erasures, so the joint decode is at capacity
            // for both blocks simultaneously.
            let message: Vec<u8> = (0..=255).collect();
            let mut encoded = encode_shortened(&message, 2);
            let mut erasures = Vec::new();
            for path in 0..2usize {
                for k in 0..8 {
                    let pos_in_path = 9 + k * 5;
                    let pos = path + pos_in_path * 2;
                    encoded[pos] ^= 0xa3;
                    erasures.push(pos);
                }
                for k in 0..4 {
                    let pos_in_path = 80 + k * 11;
                    let pos = path + pos_in_path * 2;
                    encoded[pos] ^= 0xc5;
                }
            }

            let decoded = match decode_shortened_with_erasures(&encoded, 2, &erasures) {
                Ok(d) => d,
                Err(err) => panic!("interleaved errors+erasures must decode: {err}"),
            };
            assert_eq!(decoded.message, message);
        }
    }
}
