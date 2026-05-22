//! Frame-format decoders for OpenHoshimi.
//!
//! Decoders here turn the raw bytes of a [`openhoshimi_core::Frame`] into
//! a structured representation specific to one framing protocol
//! (AX.25, AO-40 FEC, ...).

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod ax25;

pub use ax25::{Ax25Decoder, Ax25Frame, Callsign};
