//! Frame-format decoders for OpenHoshimi.
//!
//! Decoders here turn the raw bytes of a [`openhoshimi_core::Frame`] into
//! a structured representation specific to one framing protocol
//! (AX.25, AO-40 FEC, ...).

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod ao40;
pub mod ax100;
pub mod ax25;
pub mod fec;

pub use ao40::{ao40_syncword_bits, Ao40FecDecoder, Ao40Frame};
pub use ax100::{ax100_syncword, Ax100Decoder, Ax100Flags, Ax100Frame, Ax100Mode};
pub use ax25::{Ax25Decoder, Ax25Frame, Callsign};
pub use fec::ReedSolomon;
