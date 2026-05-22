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
}

impl Fec for ReedSolomon {
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>, DecodeError> {
        Ok(self.decode_shortened(data)?.message)
    }
}

pub(crate) mod ccsds_randomizer {
    const POLY_MASK: u8 = 0xa9;
    const INITIAL_STATE: u8 = 0xff;

    pub(crate) fn xor_sequence(bytes: &mut [u8]) {
        let mut state = INITIAL_STATE;
        for byte in bytes {
            let mut mask = 0u8;
            for bit_index in 0..8 {
                let pn = state & 0x80 != 0;
                if pn {
                    mask |= 1 << (7 - bit_index);
                }
                let feedback = (state & POLY_MASK).count_ones() & 1 != 0;
                state = (state << 1) | u8::from(feedback);
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

    #[cfg(test)]
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
        if interleave == 0 || codeword.len() / interleave * interleave != codeword.len() {
            return Err(DecodeError::InvalidEncoding(
                "invalid Reed-Solomon interleave".to_string(),
            ));
        }

        let rs_nn = codeword.len() / interleave;
        if rs_nn <= PARITY_LEN || rs_nn > NN {
            return Err(DecodeError::TooShort(codeword.len()));
        }

        let pad = NN - rs_nn;
        let mut corrected_errors = 0usize;
        let mut out = vec![0u8; codeword.len() - interleave * PARITY_LEN];

        for path in 0..interleave {
            let mut block = [0u8; NN];
            for k in 0..rs_nn {
                block[pad + k] = codeword[path + k * interleave];
            }

            corrected_errors += decode_block(&mut block, pad)?;

            for k in 0..(rs_nn - PARITY_LEN) {
                out[path + k * interleave] = block[pad + k];
            }
        }

        Ok(RsDecoded {
            message: out,
            corrected_errors,
        })
    }

    #[cfg(test)]
    pub(crate) fn encode_shortened(message: &[u8], interleave: usize) -> Vec<u8> {
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

    fn decode_block(data: &mut [u8; NN], pad: usize) -> Result<usize, DecodeError> {
        let mut syndromes = [0u8; PARITY_LEN];
        for (i, syndrome) in syndromes.iter_mut().enumerate() {
            *syndrome = data[pad];
            for &symbol in data.iter().skip(pad + 1) {
                if *syndrome == 0 {
                    *syndrome = symbol;
                } else {
                    let exponent =
                        usize::from(index_of(*syndrome)) + ((FCR + i as i32) * PRIM) as usize;
                    *syndrome = symbol ^ alpha_to(mod_nn(exponent));
                }
            }
        }

        let has_error = syndromes.iter().any(|syndrome| *syndrome != 0);
        if !has_error {
            return Ok(0);
        }

        Err(DecodeError::CrcMismatch)
    }

    #[cfg(test)]
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
        fn shortened_rejects_errors() {
            let message: Vec<u8> = (0..160).collect();
            let mut encoded = encode_shortened(&message, 1);
            encoded[3] ^= 0x55;

            let err = match decode_shortened(&encoded, 1) {
                Ok(_) => panic!("damaged RS codeword should fail"),
                Err(err) => err,
            };

            assert!(matches!(err, DecodeError::CrcMismatch));
        }
    }
}
