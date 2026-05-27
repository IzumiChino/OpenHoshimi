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
pub mod geoscan;
pub mod image;
pub mod pn9;

pub use ao40::{ao40_syncword_bits, Ao40FecDecoder, Ao40FecEncoder, Ao40Frame};
pub use ax100::{ax100_syncword, Ax100Decoder, Ax100Flags, Ax100Frame, Ax100Mode};
pub use ax25::{Ax25Decoder, Ax25Frame, Callsign};
pub use fec::{ReedSolomon, Viterbi};
pub use geoscan::{
    GeoscanDecoder, GeoscanFrame, GEOSCAN_CRC_LEN, GEOSCAN_FRAME_LEN, GEOSCAN_PAYLOAD_LEN,
};
pub use image::{
    build as build_image_reassembler, GeoscanImageReassembler, ImageGroup, ImageReassembler,
    ImageSnapshot, PacketUpdate, PixelFormat,
};
pub use pn9::Pn9Whitener;
