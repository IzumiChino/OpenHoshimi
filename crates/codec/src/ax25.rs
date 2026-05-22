//! AX.25 frame decoder.
//!
//! AX.25 encodes each address as six left-shifted ASCII callsign bytes
//! followed by one SSID byte. This decoder consumes an HDLC payload after
//! bit-unstuffing and FCS stripping, and returns a structured AX.25 frame.

use openhoshimi_core::{DecodeError, Frame};

const ADDRESS_LEN: usize = 7;
const MIN_ADDRESSES_LEN: usize = ADDRESS_LEN * 2;
const UI_CONTROL: u8 = 0x03;

/// AX.25 payload decoder.
#[derive(Debug, Default, Clone, Copy)]
pub struct Ax25Decoder;

impl Ax25Decoder {
    /// Construct a new stateless AX.25 decoder.
    pub fn new() -> Self {
        Self
    }

    /// Decode an OpenHoshimi [`Frame`] as an AX.25 frame.
    ///
    /// The input frame must already have had HDLC flags, bit stuffing, and
    /// FCS removed by the framing stage.
    pub fn decode_frame(&self, frame: &Frame) -> Result<Ax25Frame, DecodeError> {
        self.decode(&frame.raw)
    }

    /// Decode a raw AX.25 payload.
    ///
    /// The payload starts with destination and source address fields,
    /// optional digipeaters, then control/PID/info bytes.
    pub fn decode(&self, payload: &[u8]) -> Result<Ax25Frame, DecodeError> {
        if payload.len() < MIN_ADDRESSES_LEN {
            return Err(DecodeError::TooShort(payload.len()));
        }

        let destination = Callsign::decode(&payload[0..ADDRESS_LEN], false)?;
        let source = Callsign::decode(&payload[ADDRESS_LEN..MIN_ADDRESSES_LEN], false)?;
        let mut offset = MIN_ADDRESSES_LEN;
        let mut digipeaters = Vec::new();

        if payload[ADDRESS_LEN * 2 - 1] & 0x01 == 0 {
            loop {
                if payload.len() < offset + ADDRESS_LEN {
                    return Err(DecodeError::TooShort(payload.len()));
                }

                let address = &payload[offset..offset + ADDRESS_LEN];
                let last = address[6] & 0x01 != 0;
                digipeaters.push(Callsign::decode(address, true)?);
                offset += ADDRESS_LEN;

                if last {
                    break;
                }
            }
        }

        if payload.len() <= offset {
            return Err(DecodeError::TooShort(payload.len()));
        }

        let control = payload[offset];
        offset += 1;

        let pid = if control == UI_CONTROL {
            if payload.len() <= offset {
                return Err(DecodeError::TooShort(payload.len()));
            }
            let pid = payload[offset];
            offset += 1;
            Some(pid)
        } else {
            None
        };

        Ok(Ax25Frame {
            destination,
            source,
            digipeaters,
            control,
            pid,
            info: payload[offset..].to_vec(),
        })
    }
}

/// Decoded AX.25 frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ax25Frame {
    /// Destination address.
    pub destination: Callsign,
    /// Source address.
    pub source: Callsign,
    /// Optional digipeater path.
    pub digipeaters: Vec<Callsign>,
    /// AX.25 control byte.
    pub control: u8,
    /// Protocol ID byte, present for UI frames.
    pub pid: Option<u8>,
    /// Information field bytes.
    pub info: Vec<u8>,
}

/// AX.25 callsign and SSID pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Callsign {
    /// Callsign text, trimmed of AX.25 space padding and converted to ASCII
    /// uppercase.
    pub call: String,
    /// Secondary station identifier, in the range `0..=15`.
    pub ssid: u8,
    /// Whether a digipeater has repeated this frame.
    pub repeated: bool,
}

impl Callsign {
    fn decode(bytes: &[u8], is_digipeater: bool) -> Result<Self, DecodeError> {
        if bytes.len() != ADDRESS_LEN {
            return Err(DecodeError::TooShort(bytes.len()));
        }

        let mut call = String::new();
        for &byte in &bytes[..6] {
            let decoded = byte >> 1;
            if !(decoded == b' ' || decoded.is_ascii_alphanumeric()) {
                return Err(DecodeError::InvalidEncoding(format!(
                    "invalid AX.25 callsign byte 0x{byte:02x}"
                )));
            }
            call.push(char::from(decoded.to_ascii_uppercase()));
        }

        Ok(Self {
            call: call.trim_end().to_string(),
            ssid: (bytes[6] >> 1) & 0x0f,
            repeated: is_digipeater && bytes[6] & 0x80 != 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_call(call: &str, ssid: u8, last: bool, repeated: bool) -> [u8; ADDRESS_LEN] {
        let mut out = [b' ' << 1; ADDRESS_LEN];
        for (index, byte) in call.bytes().take(6).enumerate() {
            out[index] = byte.to_ascii_uppercase() << 1;
        }
        out[6] = 0x60 | ((ssid & 0x0f) << 1) | u8::from(last);
        if repeated {
            out[6] |= 0x80;
        }
        out
    }

    #[test]
    fn decodes_known_ui_frame() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&encode_call("ALL", 0, false, false));
        payload.extend_from_slice(&encode_call("GB1FUN", 0, true, false));
        payload.push(0x03);
        payload.push(0xf0);
        payload.extend_from_slice(b"hello");

        let decoder = Ax25Decoder::new();
        let frame = match decoder.decode(&payload) {
            Ok(frame) => frame,
            Err(err) => panic!("decode UI frame: {err}"),
        };

        assert_eq!(frame.destination.call, "ALL");
        assert_eq!(frame.destination.ssid, 0);
        assert_eq!(frame.source.call, "GB1FUN");
        assert_eq!(frame.source.ssid, 0);
        assert_eq!(frame.control, 0x03);
        assert_eq!(frame.pid, Some(0xf0));
        assert_eq!(frame.info, b"hello");
    }

    #[test]
    fn frame_shorter_than_two_addresses_is_rejected() {
        let decoder = Ax25Decoder::new();
        let err = match decoder.decode(&[0u8; 13]) {
            Ok(_) => panic!("short frame should fail"),
            Err(err) => err,
        };
        assert!(matches!(err, DecodeError::TooShort(13)));
    }
}
