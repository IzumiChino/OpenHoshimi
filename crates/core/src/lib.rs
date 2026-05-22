//! Core traits, data models, and error types for OpenHoshimi.
//!
//! This crate is the foundation of the workspace: every other crate depends
//! on it, while it itself depends on nothing from the rest of the workspace.
//! Three things live here:
//!
//! * The trait surface that defines the signal-processing pipeline
//!   ([`InputSource`] -> [`Demodulator`] -> [`Framing`] -> [`TelemetrySchema`]).
//! * The shared data structures that flow between those stages
//!   ([`Frame`], [`TelemetryField`], ...).
//! * The TOML-driven satellite definition format (see [`satellite`]).
//!
//! No satellite-specific logic is allowed in this crate or any other Rust
//! crate in the workspace - that lives entirely in `satellites/*.toml`.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod satellite;

use std::time::SystemTime;

/// Converts a stream of raw `f32` audio samples into a stream of soft bits.
///
/// Implementations are stateful - call [`push_samples`](Self::push_samples)
/// repeatedly as samples arrive. Each invocation may emit zero or more bits
/// depending on how much of a bit period has accumulated internally. Each
/// emitted byte is `0x00` or `0x01`, representing one recovered bit.
pub trait Demodulator: Send {
    /// Push a chunk of audio samples through the demodulator and return any
    /// bits that were recovered.
    ///
    /// The returned `Vec<u8>` contains one byte per recovered bit, valued
    /// `0x00` or `0x01`. The vector may be empty if not enough samples have
    /// been buffered to complete a bit yet.
    fn push_samples(&mut self, samples: &[f32]) -> Vec<u8>;

    /// Audio sample rate this demodulator was constructed for, in Hz.
    fn sample_rate(&self) -> u32;

    /// Symbol rate of the demodulated signal, in baud.
    fn baudrate(&self) -> u32;
}

/// Finds frame boundaries in a bit (or byte) stream and returns complete
/// frames.
///
/// The exact interpretation of the input bytes depends on the implementation:
/// some framers (HDLC) take a one-bit-per-byte stream from a [`Demodulator`],
/// others may take packed bytes. Each implementation documents its expected
/// input format.
pub trait Framing: Send {
    /// Push bytes through the framer and return any complete frames that
    /// were recovered as a result. The vector may be empty.
    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<Frame>;
}

/// Parses a raw [`Frame`] into human-readable telemetry fields.
///
/// Implementations are typically constructed from a parsed satellite TOML
/// definition (see [`satellite::TelemetrySchemaDef`]) and used as a stateless
/// transform.
pub trait TelemetrySchema: Send {
    /// Parse the raw bytes of a frame into a list of telemetry fields.
    fn parse(&self, frame: &Frame) -> Vec<TelemetryField>;
}

/// Provides a stream of `f32` audio samples from any source (WAV file, IQ
/// file, soundcard, ...).
pub trait InputSource: Send {
    /// Fill `buf` with up to `buf.len()` samples and return how many were
    /// actually written. Returning `0` is allowed for non-blocking sources
    /// that have nothing buffered. To signal end of stream, return
    /// [`IoError::EndOfStream`].
    fn read_samples(&mut self, buf: &mut [f32]) -> Result<usize, IoError>;

    /// Sample rate of the produced audio, in Hz.
    fn sample_rate(&self) -> u32;

    /// Human-readable description of the source, e.g. `"WAV file
    /// recording.wav (48000 Hz, 16-bit, mono)"`.
    fn description(&self) -> &str;
}

/// A complete decoded frame, after framing but before telemetry parsing.
///
/// `raw` holds the payload bytes only - flag/preamble bytes and CRCs are
/// stripped by the [`Framing`] stage.
#[derive(Debug, Clone)]
pub struct Frame {
    /// NORAD catalog number of the satellite this frame is attributed to.
    pub satellite_id: u32,
    /// Wall-clock time at which the frame was received.
    pub timestamp: SystemTime,
    /// Optional received signal strength indicator at the time of reception.
    pub rssi_dbm: Option<f32>,
    /// Raw payload bytes of the frame.
    pub raw: Vec<u8>,
    /// Framing protocol that produced this frame.
    pub frame_type: FrameType,
}

/// The framing protocol a [`Frame`] was decoded from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Standard amateur-radio AX.25 frame (typically over HDLC).
    Ax25,
    /// AMSAT AO-40 forward-error-correction frame format.
    Ao40Fec,
    /// GOMspace AX100 transceiver frame format.
    GomspaceAx100,
    /// CCSDS space-data-link frame.
    Ccsds,
    /// FX.25 (AX.25 wrapped in a Reed-Solomon outer code).
    Fx25,
    /// Frame whose framing is unknown or not yet classified.
    Unknown,
}

/// A single telemetry datapoint extracted from a [`Frame`].
#[derive(Debug, Clone)]
pub struct TelemetryField {
    /// Short identifier of the field (e.g. `"bat_voltage"`).
    pub key: String,
    /// Group/category the field belongs to (e.g. `"eps"`, `"thermal"`).
    pub group: String,
    /// Decoded value of the field.
    pub value: TelemetryValue,
    /// Engineering unit of the field, if any (e.g. `"V"`, `"C"`).
    pub unit: Option<String>,
    /// Whether the field is in a healthy range, given thresholds defined in
    /// the satellite TOML.
    pub warn: WarnLevel,
}

/// The decoded value of a [`TelemetryField`].
#[derive(Debug, Clone, PartialEq)]
pub enum TelemetryValue {
    /// A scaled, dimensionful floating-point value.
    Float(f64),
    /// A signed integer value.
    Int(i64),
    /// A boolean value (e.g. a status flag).
    Bool(bool),
    /// Raw bytes - used for fields whose interpretation is opaque.
    Bytes(Vec<u8>),
}

/// Severity of a telemetry-field warning, derived from `warn_below` /
/// `warn_above` thresholds in the satellite TOML.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarnLevel {
    /// The value is within all configured thresholds.
    Ok,
    /// The value crossed a soft threshold but is not yet critical.
    Warn,
    /// The value is outside any reasonable operating range.
    Error,
}

/// Errors emitted by [`InputSource`] implementations and other I/O code.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// Underlying [`std::io`] error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// File or stream format is not understood (e.g. unsupported WAV
    /// sample format, malformed header).
    #[error("Format error: {0}")]
    Format(String),
    /// The end of the input stream was reached.
    #[error("End of stream")]
    EndOfStream,
}

/// Errors emitted by demodulator/framer/decoder code while processing a
/// frame.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Frame failed its CRC check.
    #[error("CRC mismatch")]
    CrcMismatch,
    /// Frame is shorter than the minimum length for its protocol.
    #[error("Frame too short: {0} bytes")]
    TooShort(usize),
    /// A field encoded inside a frame uses an encoding the decoder cannot
    /// understand.
    #[error("Invalid encoding: {0}")]
    InvalidEncoding(String),
}

/// Errors emitted while loading or validating a satellite TOML definition.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Underlying [`std::io`] error (file could not be opened, read, ...).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// The TOML payload is syntactically invalid.
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    /// A required field was missing from the TOML.
    #[error("Missing field: {0}")]
    MissingField(String),
    /// A field was present but contained an invalid value.
    #[error("Invalid value for {field}: {reason}")]
    InvalidValue {
        /// Name of the offending field.
        field: String,
        /// Human-readable reason the value was rejected.
        reason: String,
    },
}
