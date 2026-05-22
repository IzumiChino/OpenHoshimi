//! Input sources for OpenHoshimi.
//!
//! Currently provides [`wav::WavSource`] and [`wav::WavIqSource`] for
//! offline file decoding. Live soundcard and IQ-file sources will be added
//! later.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod wav;

pub use wav::{WavIqSource, WavSource};
