//! Demodulators and bit-level processing for OpenHoshimi.
//!
//! See [`afsk::AfskDemodulator`] for Bell 202 audio FSK,
//! [`ao40::Ao40Framer`] for AO-40 FEC distributed sync framing,
//! [`ccsds::CcsdsDescrambler`] for CCSDS randomizer descrambling,
//! [`cpm::CpmDemodulator`] for FSK/MSK/GFSK/GMSK IQ streams,
//! [`fm_audio::FmAudioDemodulator`] for FM-demodulated audio symbol recovery,
//! [`hilbert::HilbertTransform`] for real-to-analytic signal conversion,
//! [`linear::LinearDemodulator`] for BPSK/DBPSK/QPSK/OQPSK IQ streams,
//! [`g3ruh::G3ruhDemodulator`] for 9600 baud packet-radio bit processing,
//! [`hdlc::HdlcFramer`] for AX.25-style HDLC framing, and
//! [`syncword::SyncwordFramer`] for fixed-sync protocols.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod afsk;
pub mod ao40;
pub mod ccsds;
pub mod cpm;
pub mod fft;
pub mod fm_audio;
pub mod g3ruh;
pub mod hdlc;
pub mod hilbert;
pub mod linear;
pub mod syncword;

pub use afsk::AfskDemodulator;
pub use ao40::Ao40Framer;
pub use ccsds::CcsdsDescrambler;
pub use cpm::{CpmConfig, CpmDemodulator, CpmMode};
pub use fft::{bin_frequency_hz, fft_in_place, hann_window, Complex};
pub use fm_audio::{FmAudioConfig, FmAudioDemodulator};
pub use g3ruh::{G3ruhDemodulator, G3ruhDescrambler, NrziDecoder};
pub use hdlc::HdlcFramer;
pub use hilbert::HilbertTransform;
pub use linear::{LinearConfig, LinearDemodulator, LinearMode};
pub use syncword::{pack_msb_bits, SyncwordFramer};
