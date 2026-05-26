//! Input sources for OpenHoshimi.
//!
//! This crate provides offline WAV readers, OGG Vorbis readers, and a live
//! soundcard input source.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod ogg;
pub mod soundcard;
pub mod wav;

pub use ogg::OggSource;
pub use soundcard::SoundcardSource;
pub use wav::{WavIqSource, WavSource};
