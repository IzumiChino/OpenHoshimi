//! Demodulators and bit-level processing for OpenHoshimi.
//!
//! See [`afsk::AfskDemodulator`] for Bell 202 audio FSK and
//! [`hdlc::HdlcFramer`] for AX.25-style HDLC framing.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod afsk;
pub mod hdlc;

pub use afsk::AfskDemodulator;
pub use hdlc::HdlcFramer;
