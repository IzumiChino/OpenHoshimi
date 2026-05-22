//! Input sources for OpenHoshimi.
//!
//! Currently provides [`wav::WavSource`] for offline file decoding. Live
//! soundcard and IQ-file sources will be added later.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod wav;

pub use wav::WavSource;
