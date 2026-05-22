//! Input sources for OpenHoshimi.
//!
//! Currently provides [`wav::WavSource`] for offline file decoding. Live
//! soundcard and IQ-file sources arrive in Phase 4.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod wav;

pub use wav::WavSource;
