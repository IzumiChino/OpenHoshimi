//! Input sources for OpenHoshimi.
//!
//! This crate provides offline WAV readers, OGG Vorbis readers, a live
//! soundcard input source, and the mono-real-audio adapters used by the
//! FM/SSB audio paths.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod mono_iq;
pub mod ogg;
pub mod soundcard;
pub mod wav;

pub use mono_iq::{
    detect_audio_mode_auto, is_duplicate_stereo, open_audio_source, read_audio_prefix,
    read_iq_prefix, AudioMode, MonoIqSource,
};
pub use ogg::OggSource;
pub use soundcard::{
    enumerate_input_devices, SoundcardDeviceInfo, SoundcardIqSource, SoundcardSource,
};
pub use wav::{WavIqSource, WavSource};
