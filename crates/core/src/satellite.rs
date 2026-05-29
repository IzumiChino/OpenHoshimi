//! TOML-driven satellite definition format and loader.
//!
//! Everything that distinguishes one amateur satellite from another - its
//! NORAD ID, its downlink frequencies, the modulation and framing of each
//! downlink, and the layout of the telemetry inside each framing - lives in
//! a `*.toml` file under `satellites/`. The Rust types in this module are
//! [`serde::Deserialize`] mirrors of that TOML schema; nothing in here is
//! satellite-specific.
//!
//! The canonical schema is documented on [`SatelliteDefinition`].

use std::collections::HashMap;
use std::path::Path;

use serde::de::Error as _;
use serde::Deserialize;

use crate::ConfigError;

/// Top-level structure of a satellite TOML file.
///
/// # Schema
///
/// ```toml
/// [satellite]
/// name = "AO-73"
/// aliases = ["FUNcube-1"]
/// norad_id = 39444
///
/// [[downlink]]
/// label = "1k2 BPSK beacon"
/// freq_hz = 145_935_000
/// modulation = "DBPSK"
/// baudrate = 1200
/// framing = "AO40_FEC"
/// telemetry_schema = "funcube_1"
///
/// [telemetry.funcube_1]
///
/// [[telemetry.funcube_1.field]]
/// name = "bat_voltage"
/// group = "eps"
/// offset = 2
/// length = 2
/// endian = "big"
/// scale = 0.001
/// unit = "V"
/// warn_below = 6.5
/// warn_above = 8.4
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct SatelliteDefinition {
    /// Identifying section: name, aliases, NORAD ID.
    #[serde(deserialize_with = "required")]
    pub satellite: SatelliteHeader,
    /// One entry per distinct downlink the satellite transmits.
    #[serde(default, rename = "downlink")]
    pub downlinks: Vec<DownlinkDef>,
    /// Map from `telemetry_schema` name (referenced by a downlink) to its
    /// field layout.
    #[serde(default)]
    pub telemetry: HashMap<String, TelemetrySchemaDef>,
}

/// Identifying header of a satellite definition.
#[derive(Debug, Clone, Deserialize)]
pub struct SatelliteHeader {
    /// Primary display name (e.g. `"AO-73"`).
    #[serde(deserialize_with = "required")]
    pub name: String,
    /// Optional alternative names (e.g. `["FUNcube-1"]`).
    #[serde(default)]
    pub aliases: Vec<String>,
    /// NORAD catalog number, used as the satellite ID on decoded frames.
    #[serde(deserialize_with = "required")]
    pub norad_id: u32,
}

/// One downlink (a single carrier on a single frequency) of a satellite.
#[derive(Debug, Clone, Deserialize)]
pub struct DownlinkDef {
    /// Short human-readable label (e.g. `"1k2 BPSK beacon"`).
    #[serde(deserialize_with = "required")]
    pub label: String,
    /// Centre frequency in Hertz.
    #[serde(deserialize_with = "required")]
    pub freq_hz: u64,
    /// Modulation, e.g. `"DBPSK"`, `"AFSK"`, `"GMSK"`, `"FSK"`.
    #[serde(deserialize_with = "required")]
    pub modulation: String,
    /// Symbol rate in baud.
    #[serde(deserialize_with = "required")]
    pub baudrate: u32,
    /// Framing protocol, e.g. `"AX25"`, `"AO40_FEC"`, `"GOMSPACE_AX100"`.
    #[serde(deserialize_with = "required")]
    pub framing: String,
    /// Name of a telemetry schema in [`SatelliteDefinition::telemetry`].
    /// Optional - some downlinks (e.g. voice repeaters) carry no telemetry.
    #[serde(default)]
    pub telemetry_schema: Option<String>,
    /// Structured modem configuration for this downlink.
    #[serde(default)]
    pub modem: Option<ModemDef>,
    /// Optional line coding transform after demodulation.
    #[serde(default)]
    pub line_coding: Option<LineCodingDef>,
    /// Optional bit descrambler after line decoding.
    #[serde(default)]
    pub descrambler: Option<DescramblerDef>,
    /// Structured framer configuration.
    #[serde(default)]
    pub framer: Option<FramerDef>,
    /// Optional forward-error-correction stage.
    #[serde(default)]
    pub fec: Option<FecDef>,
    /// Structured frame decoder configuration.
    #[serde(default)]
    pub codec: Option<CodecDef>,
    /// Optional image-frame reassembly stage. Set when the downlink
    /// carries images split across many small frames (e.g. STRATOSAT-TK-1
    /// thumbnails).
    #[serde(default)]
    pub image: Option<ImageDef>,
}

/// Modem configuration for a downlink.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModemDef {
    /// Bell 202 audio AFSK.
    Afsk {
        /// Mark tone frequency in Hz.
        #[serde(default = "default_afsk_mark_hz")]
        mark_hz: f32,
        /// Space tone frequency in Hz.
        #[serde(default = "default_afsk_space_hz")]
        space_hz: f32,
    },
    /// Binary continuous-phase modulation over IQ.
    Cpm {
        /// CPM family mode.
        mode: CpmModeDef,
        /// Modulation index.
        #[serde(default)]
        modulation_index: Option<f32>,
        /// Frequency offset correction in Hz.
        #[serde(default)]
        frequency_offset_hz: f32,
        /// Gaussian BT product for GFSK/GMSK.
        #[serde(default)]
        gaussian_bt: Option<f32>,
        /// Decode differential encoding after hard slicing.
        #[serde(default)]
        differential: bool,
        /// Invert hard symbol decisions.
        #[serde(default)]
        invert: bool,
        /// Swap I and Q before demodulation.
        #[serde(default)]
        swap_iq: bool,
    },
    /// Linear phase modulation over IQ.
    Linear {
        /// Linear modulation mode.
        mode: LinearModeDef,
        /// Frequency offset correction in Hz, applied before symbol slicing.
        #[serde(default)]
        frequency_offset_hz: f32,
        /// Decode differential encoding after hard slicing.
        #[serde(default)]
        differential: bool,
        /// Invert hard symbol decisions.
        #[serde(default)]
        invert: bool,
        /// Swap I and Q before demodulation.
        #[serde(default)]
        swap_iq: bool,
        /// Closed-loop carrier tracker bandwidth normalized to the symbol
        /// rate. `0.0` keeps the static frequency correction only.
        #[serde(default)]
        carrier_loop_bandwidth: f32,
        /// Frequency-locked loop bandwidth normalized to the symbol rate.
        /// Drives a Kay-style cross-product discriminator on squared symbols
        /// for wide pull-in over Doppler. `0.0` disables the FLL and leaves
        /// only the phase-locked Costas tracker active.
        #[serde(default)]
        frequency_loop_bandwidth: f32,
        /// Maximum absolute offset the NCO is allowed to slew from the
        /// static `frequency_offset_hz` baseline, in Hz. `0.0` selects a
        /// conservative default of one symbol rate. Increase this for LEO
        /// passes whose Doppler swing exceeds the default.
        #[serde(default)]
        nco_max_offset_hz: f32,
        /// Root-raised-cosine matched filter roll-off factor in `(0.0, 1.0]`.
        /// `0.0` disables matched filtering and falls back to a boxcar
        /// integrate-and-dump symbol filter.
        #[serde(default)]
        matched_filter_rolloff: f32,
        /// Matched filter span in symbol periods. Used only when
        /// `matched_filter_rolloff > 0`. `0` selects a default (six symbols).
        #[serde(default)]
        matched_filter_span_symbols: usize,
    },
    /// LoRa modem placeholder.
    Lora {
        /// LoRa spreading factor.
        spreading_factor: u8,
        /// LoRa bandwidth in Hz.
        bandwidth_hz: u32,
    },
    /// Four-level FSK modem placeholder.
    FourFsk {
        /// Frequency offsets for the four symbols, in Hz.
        freq_offsets_hz: [f32; 4],
    },
}

/// CPM modulation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpmModeDef {
    /// Binary FSK.
    Fsk,
    /// Minimum-shift keying.
    Msk,
    /// Gaussian FSK.
    Gfsk,
    /// Gaussian MSK.
    Gmsk,
}

/// Linear modulation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearModeDef {
    /// Binary PSK.
    Bpsk,
    /// Differential binary PSK.
    Dbpsk,
    /// Quadrature PSK.
    Qpsk,
    /// Offset quadrature PSK.
    Oqpsk,
}

/// Line coding transform configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LineCodingDef {
    /// NRZI line coding.
    Nrzi,
    /// NRZ-S line coding placeholder.
    Nrzs,
    /// NRZ-M line coding placeholder.
    Nrzm,
}

/// Descrambler configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DescramblerDef {
    /// G3RUH self-synchronising descrambler.
    G3ruh,
    /// CCSDS randomizer descrambler.
    Ccsds,
}

/// Framer configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FramerDef {
    /// HDLC flag-based framing.
    Hdlc,
    /// AO-40 distributed sync framing.
    Ao40 {
        /// Maximum number of bit errors allowed in the distributed sync
        /// vector.
        #[serde(default)]
        threshold: usize,
    },
    /// Fixed syncword followed by a fixed-size payload.
    Syncword {
        /// Syncword bits as ASCII `0` and `1` characters.
        syncword: String,
        /// Maximum number of bit errors allowed in the syncword.
        #[serde(default)]
        threshold: usize,
        /// Number of payload bits to collect after syncword detection.
        payload_bits: usize,
    },
    /// GOMspace AX100 ASM-prefixed framing. Hunts the 32-bit attached sync
    /// marker `0x930b_51de`, then emits a fixed 258-byte (3-byte Golay header
    /// plus 255-byte CCSDS-randomised RS codeword) payload that the AX100
    /// codec decodes.
    Ax100Asm {
        /// Maximum number of bit errors allowed in the 32-bit ASM.
        #[serde(default = "default_ax100_asm_threshold")]
        threshold: usize,
    },
}

/// Forward-error-correction stage configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FecDef {
    /// Reed-Solomon codeword decoder.
    ReedSolomon {
        /// Interleave factor.
        #[serde(default = "default_interleave")]
        interleave: usize,
    },
    /// AO-40 FEC decoder placeholder for the full chain.
    Ao40,
    /// GOMspace AX100 FEC wrapper.
    Ax100,
}

/// Frame decoder configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CodecDef {
    /// AX.25 frame decoder.
    Ax25,
    /// AO-40 FEC payload decoder.
    Ao40Fec,
    /// GOMspace AX100 decoder.
    GomspaceAx100 {
        /// AX100 decoder mode.
        mode: Ax100ModeDef,
    },
    /// CCSDS frame decoder placeholder.
    Ccsds,
    /// FX.25 frame decoder placeholder.
    Fx25,
    /// Geoscan custom frame decoder (CC11xx PN9 descrambling +
    /// CC11xx CRC16 over a fixed-size payload).
    Geoscan,
    /// SSDV (Slow-Scan Digital Video) packet decoder. Operates on
    /// 256-byte packets carrying a JPEG MCU slice with optional
    /// Reed-Solomon parity. See `crates/codec/src/ssdv.rs`.
    Ssdv,
    /// Unknown or unsupported frame decoder.
    Unknown,
}

/// GOMspace AX100 decoder mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ax100ModeDef {
    /// AX100 Reed-Solomon mode.
    ReedSolomon,
    /// AX100 ASM and Golay mode with CCSDS scrambling and RS(255,223).
    AsmGolay,
    /// AX100 ASM and Golay mode with CCSDS scrambling and a CRC-32C
    /// trailer, without Reed-Solomon. Used by the GreenCube / IO-117
    /// digipeater downlink.
    AsmGolayCrc,
}

/// Image-frame reassembly configuration.
///
/// When a downlink carries images split across many small frames, the
/// runtime hands each decoded payload to a stateful reassembler that
/// drops it into the correct slot of a per-image canvas. v1 only supports
/// the Geoscan custom protocol used by STRATOSAT-TK-1 and Geoscan-Edelveis;
/// SSDV and other protocols arrive as additional variants of this enum.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "protocol", rename_all = "snake_case")]
pub enum ImageDef {
    /// Geoscan custom protocol: each frame begins with a fixed
    /// `header_signature`, followed by a 16-bit `packet_offset` field
    /// (byte offset into the linear image buffer) and a fixed-length raw
    /// chunk. The protocol does not carry an explicit image id; new
    /// images are detected by the offset returning to zero (with the
    /// chunk starting with the `decoder`'s start-of-image marker, e.g.
    /// the JPEG `FF D8` SOI).
    Geoscan {
        /// Header bytes that must appear at the start of an image frame's
        /// payload. Whitespace is ignored. Frames whose first
        /// `header_signature.len()/2` bytes do not match are skipped by
        /// the reassembler.
        header_signature: String,
        /// Byte location, length, and endianness of the packet-offset
        /// field inside the frame payload.
        offset_field: ImageField,
        /// First byte of the raw image chunk inside the frame payload.
        chunk_at: usize,
        /// Length of the raw image chunk in bytes (typically 56 for the
        /// 64-byte Geoscan payload).
        chunk_bytes: usize,
        /// How to interpret the reassembled byte stream when rendering
        /// or saving. `Jpeg` decodes a compressed JPEG bitstream and lets
        /// the canvas size follow the image header; `Raw` lays the bytes
        /// out as raw pixels using `width`/`height`/`pixel_format`.
        /// Defaults to `Jpeg` because the only known producer
        /// (STRATOSAT-TK-1 / Geoscan-Edelveis) sends JPEG.
        #[serde(default)]
        decoder: ImageDecoderDef,
        /// Canvas width in pixels. Used only when `decoder = "raw"`.
        /// Defaults to 320.
        #[serde(default = "default_image_width")]
        width: u32,
        /// Canvas height in pixels. Used only when `decoder = "raw"`.
        /// Defaults to 240.
        #[serde(default = "default_image_height")]
        height: u32,
        /// Pixel format used to interpret raw chunk bytes. Used only
        /// when `decoder = "raw"`. Defaults to `Gray8`.
        #[serde(default)]
        pixel_format: PixelFormatDef,
    },
    /// SSDV protocol: each frame's payload is a complete 256-byte
    /// SSDV packet (sync byte, type, base-40 callsign, image_id,
    /// packet_id, geometry, payload, CRC32, optional Reed-Solomon
    /// parity). The reassembler multiplexes by `image_id`, slots
    /// payloads into per-`packet_id` buckets, and lazily builds a
    /// JPEG bitstream with placeholder MCUs for missing packets.
    Ssdv {
        /// Optional callsign filter; when set, packets whose
        /// decoded callsign does not match are ignored. Set when a
        /// downlink legitimately carries traffic from multiple
        /// senders and the operator wants to track one.
        #[serde(default)]
        callsign: Option<String>,
    },
    /// Slow-Scan Television (SSTV) audio image. Pixel intensity is
    /// encoded as audio frequency on a 1500-2300 Hz ramp; the mode
    /// (Robot36, PD120, ...) is auto-detected from the 30-bit VIS
    /// code at the start of each transmission. SSTV is fundamentally
    /// not a bit pipeline - the runtime feeds raw audio samples
    /// directly to a [`crate`-`adjacent`] `SstvAnalyzer` instead of
    /// going through demodulator / framer / codec stages, so an
    /// SSTV downlink legitimately omits all three. The presence of
    /// this `[downlink.image]` block is the signal that the SSTV
    /// runtime path should be used.
    Sstv {},
}

/// Decoder applied to the reassembled byte stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDecoderDef {
    /// Reassembled bytes are a JPEG bitstream. Image-boundary detection
    /// looks for the JPEG SOI marker (`FF D8`) in the first chunk of a
    /// stream that arrived at offset 0.
    #[default]
    Jpeg,
    /// Reassembled bytes are raw pixels in the `pixel_format` layout.
    /// Image-boundary detection falls back to "offset returned to 0
    /// after the canvas had data".
    Raw,
}

/// Location of a fixed integer field inside a frame payload.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageField {
    /// Byte offset of the field's first byte inside the frame payload.
    pub at: usize,
    /// Field length in bytes.
    pub len: usize,
    /// Byte order of multi-byte fields. Defaults to big-endian.
    #[serde(default)]
    pub endian: ByteOrderDef,
}

/// Pixel format for an image-reassembly canvas.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFormatDef {
    /// 8-bit grayscale; one byte per pixel.
    #[default]
    Gray8,
    /// 16-bit RGB565; two bytes per pixel, big-endian.
    Rgb565,
    /// 24-bit RGB888; three bytes per pixel.
    Rgb888,
}

/// Byte order for multi-byte integer fields.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ByteOrderDef {
    /// Big-endian (most significant byte first).
    #[default]
    Be,
    /// Little-endian (least significant byte first).
    Le,
}

/// A telemetry schema: the layout of fields inside a frame's payload.
///
/// Schemas are referenced by name from a [`DownlinkDef::telemetry_schema`].
///
/// Two layout shapes are supported:
///
/// * **Flat schema** — a single [`fields`](Self::fields) list applied to
///   every frame. Used by FUNcube-1 (AO-73), HADES-R, IO-117, etc.
/// * **Variant schema** — a [`discriminator`](Self::discriminator) field
///   selects one of [`variants`](Self::variants), each carrying its own
///   `fields` list. Used by satellites whose beacons multiplex several
///   subsystem layouts behind a single packet ID (LUME-1's B1..B5).
///
/// When a schema declares both top-level `fields` and `variants`, the
/// variant whose `match_value` equals the discriminator wins; if no
/// variant matches, the top-level `fields` are used as a fallback.
///
/// [`prefix_skip`](Self::prefix_skip) bytes are stripped from the front of
/// the payload before any field offsets are evaluated. Discriminator
/// offsets are evaluated against the *original* (unstripped) payload so
/// the same TOML can describe a transport-stack header that the field
/// schema otherwise wants to ignore.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TelemetrySchemaDef {
    /// Number of bytes to skip at the start of the payload before
    /// applying field offsets. Defaults to `0`. Discriminator offsets are
    /// evaluated against the unstripped payload.
    #[serde(default)]
    pub prefix_skip: usize,
    /// Whitelist of frame lengths that this schema applies to. When
    /// non-empty, the parser returns no fields for any frame whose raw
    /// length is not in the list - guarding against multi-purpose framing
    /// where the same codec emits both telemetry and non-telemetry
    /// payloads (e.g. IO-117's AX.100 link, which carries both 101-byte
    /// HK packets and store-and-forward digipeater traffic).
    #[serde(default)]
    pub match_lengths: Vec<usize>,
    /// Optional discriminator that selects one of [`variants`](Self::variants).
    #[serde(default)]
    pub discriminator: Option<DiscriminatorDef>,
    /// Per-variant field layouts, keyed by discriminator value.
    #[serde(default, rename = "variant")]
    pub variants: Vec<TelemetryVariantDef>,
    /// Top-level field layout, used when no discriminator matches (or
    /// when no discriminator is configured).
    #[serde(default, rename = "field")]
    pub fields: Vec<TelemetryFieldDef>,
}

/// Discriminator field used by a [`TelemetrySchemaDef`] to pick a variant.
///
/// The discriminator is read as an unsigned big-endian or little-endian
/// integer of `length` bytes from `offset` in the original payload (i.e.
/// before [`TelemetrySchemaDef::prefix_skip`] is applied). `length` must
/// be 1, 2, 4, or 8.
#[derive(Debug, Clone, Deserialize)]
pub struct DiscriminatorDef {
    /// Byte offset of the discriminator inside the unstripped payload.
    pub offset: usize,
    /// Discriminator width in bytes; must be 1, 2, 4, or 8.
    pub length: usize,
    /// Byte order of the discriminator value.
    #[serde(default = "default_endian")]
    pub endian: Endian,
}

/// One variant inside a [`TelemetrySchemaDef`] keyed by discriminator value.
///
/// A variant matches when the discriminator value equals
/// [`match_value`](Self::match_value) **or** appears in
/// [`match_values`](Self::match_values). Use the list form when several
/// IDs share the exact same field layout (e.g. IO-117's four legal
/// `tlm_id` values 13840 / 13841 / 13842 / 30234 all decoding the same
/// HK block).
#[derive(Debug, Clone, Deserialize)]
pub struct TelemetryVariantDef {
    /// Short identifier for the variant (e.g. `"b1_obc"`). Surfaced to
    /// the UI to label which variant produced a given decode.
    #[serde(deserialize_with = "required")]
    pub name: String,
    /// Single discriminator value that selects this variant. Defaults to
    /// `0` and is unused when [`match_values`](Self::match_values) is
    /// non-empty.
    #[serde(default)]
    pub match_value: u64,
    /// List of discriminator values that all select this variant. When
    /// non-empty, [`match_value`](Self::match_value) is ignored. Useful
    /// when several IDs share the same field layout.
    #[serde(default)]
    pub match_values: Vec<u64>,
    /// Field list applied (against the prefix-stripped payload) when
    /// this variant is selected.
    #[serde(default, rename = "field")]
    pub fields: Vec<TelemetryFieldDef>,
}

/// One telemetry field's location and decoding rule inside a frame.
///
/// A field may be addressed in two ways:
///
/// * **Byte-level** by setting [`offset`](Self::offset) and
///   [`length`](Self::length); the raw value is the unsigned integer formed
///   from those bytes in the chosen [`endian`](Self::endian) order.
/// * **Bit-level** by setting [`bit_offset`](Self::bit_offset) and
///   [`bit_length`](Self::bit_length); the payload is treated as an
///   MSB-first bit stream (bit 0 of byte 0 is the most significant bit) and
///   `bit_length` bits are extracted starting at `bit_offset`. The byte
///   `endian` is ignored for bit-level fields.
///
/// The two modes are mutually exclusive. Bit-level addressing is required
/// for densely packed schemas like FUNcube-1's RealTimeFC1 frame.
///
/// Calibration of the raw integer `r` is
/// `engineering = scale * r^power + bias`, where `power` defaults to `1`
/// (a plain linear `scale * r + bias`).
#[derive(Debug, Clone, Deserialize)]
pub struct TelemetryFieldDef {
    /// Short identifier (e.g. `"bat_voltage"`).
    #[serde(deserialize_with = "required")]
    pub name: String,
    /// Group/category the field belongs to (e.g. `"eps"`).
    #[serde(deserialize_with = "required")]
    pub group: String,
    /// Byte offset from the start of the frame payload, when addressing
    /// the field byte-by-byte. Mutually exclusive with
    /// [`bit_offset`](Self::bit_offset).
    #[serde(default)]
    pub offset: Option<usize>,
    /// Length of the raw field in bytes. Mutually exclusive with
    /// [`bit_length`](Self::bit_length).
    #[serde(default)]
    pub length: Option<usize>,
    /// Bit offset from the start of the frame payload, when addressing
    /// the field at sub-byte granularity. Mutually exclusive with
    /// [`offset`](Self::offset).
    #[serde(default)]
    pub bit_offset: Option<usize>,
    /// Length of the raw field in bits, up to 64. Mutually exclusive with
    /// [`length`](Self::length).
    #[serde(default)]
    pub bit_length: Option<usize>,
    /// Byte order of the raw integer (`"big"` or `"little"`).
    /// Defaults to big-endian, matching the AMSAT/CCSDS convention.
    /// Ignored for bit-level fields, which are always assembled MSB-first.
    #[serde(default = "default_endian")]
    pub endian: Endian,
    /// Multiplicative scale: `engineering = scale * raw^power + bias`.
    /// Defaults to `1.0`.
    #[serde(default = "default_scale")]
    pub scale: f64,
    /// Additive bias applied after the scale and power. Defaults to `0.0`.
    #[serde(default)]
    pub bias: f64,
    /// Power exponent applied to the raw value before `scale`. Defaults to
    /// `1.0` (plain linear). Used by FUNcube-1 RF-power telemetry which is
    /// calibrated as `5e-3 * raw^2.0629`.
    #[serde(default = "default_power")]
    pub power: f64,
    /// Engineering unit (`"V"`, `"C"`, ...), if any.
    #[serde(default)]
    pub unit: Option<String>,
    /// Soft lower threshold below which the field is considered abnormal.
    #[serde(default)]
    pub warn_below: Option<f64>,
    /// Soft upper threshold above which the field is considered abnormal.
    #[serde(default)]
    pub warn_above: Option<f64>,
    /// If true, the raw byte value is interpreted as a two's-complement
    /// signed integer of the same width before scale/bias are applied.
    /// Defaults to `false` (unsigned). Only meaningful for byte-level
    /// fields whose [`encoding`](Self::encoding) is the default integer
    /// encoding; ignored for floats and custom encodings.
    #[serde(default)]
    pub signed: bool,
    /// Custom raw-value encoding. Defaults to plain unsigned/signed
    /// integer (per the [`signed`](Self::signed) flag). Floats and
    /// satellite-specific encodings are selected here.
    #[serde(default)]
    pub encoding: FieldEncoding,
    /// Number of decimal digits encoded in the fractional byte of a
    /// [`FieldEncoding::BcdSplit`] field. Defaults to `1` (e.g. byte
    /// pair `0x0C 0x03` = 12.3). Ignored for other encodings.
    #[serde(default = "default_decimal_places")]
    pub decimal_places: u8,
}

/// Raw-value encoding for a telemetry field.
///
/// The default [`Integer`](Self::Integer) encoding pulls a 1/2/4/8-byte
/// integer in the field's byte order, then applies the field's
/// `scale * raw^power + bias` calibration. Other encodings reinterpret
/// the bytes before calibration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldEncoding {
    /// Plain unsigned (or signed when [`TelemetryFieldDef::signed`] is
    /// set) big/little-endian integer. Default.
    #[default]
    Integer,
    /// IEEE-754 binary32 in the field's byte order. `length` must be 4.
    Float32,
    /// IEEE-754 binary64 in the field's byte order. `length` must be 8.
    Float64,
    /// CAMSAT BCD-split: byte 0 is the integer part, byte 1 is the
    /// fractional part interpreted as decimal digits. The number of
    /// decimal digits is given by the field's
    /// [`decimal_places`](TelemetryFieldDef::decimal_places); the
    /// engineering value is `b0 + b1 / 10^decimal_places`. `length` must
    /// be 2.
    BcdSplit,
    /// 1-byte sign-magnitude temperature (CAS-5A): bit 7 is the sign
    /// (1 = negative), bits 6..0 are the magnitude. `length` must be 1.
    SignMagnitude8,
    /// 2-byte signed Q15 quaternion component (CAS-5A): the bytes are
    /// `[low, high]` in the frame, the assembled int16 is divided by
    /// `32768.0`. `length` must be 2.
    Q15Signed,
}

/// Byte order tag for a telemetry field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Endian {
    /// Most significant byte first.
    Big,
    /// Least significant byte first.
    Little,
}

fn default_endian() -> Endian {
    Endian::Big
}

fn default_scale() -> f64 {
    1.0
}

fn default_power() -> f64 {
    1.0
}

fn default_decimal_places() -> u8 {
    1
}

fn default_afsk_mark_hz() -> f32 {
    1200.0
}

fn default_afsk_space_hz() -> f32 {
    2200.0
}

fn default_interleave() -> usize {
    1
}

fn default_ax100_asm_threshold() -> usize {
    4
}

fn default_image_width() -> u32 {
    320
}

fn default_image_height() -> u32 {
    240
}

fn required<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)?
        .ok_or_else(|| D::Error::custom("missing required field"))
}

/// Load a single satellite definition from a `*.toml` file.
///
/// # Errors
///
/// Returns [`ConfigError::Io`] if the file cannot be read,
/// [`ConfigError::Toml`] if the contents are not valid TOML, or
/// [`ConfigError::MissingField`] / [`ConfigError::InvalidValue`] if a
/// required field is absent or invalid.
pub fn load_satellite(path: &Path) -> Result<SatelliteDefinition, ConfigError> {
    let raw = std::fs::read_to_string(path)?;
    let def: SatelliteDefinition = toml::from_str(&raw).map_err(translate_toml_error)?;
    validate(&def)?;
    Ok(def)
}

/// Load every `*.toml` file in a directory.
///
/// Files that fail to parse are reported on stderr and skipped - the
/// returned vector contains only the successfully parsed definitions. This
/// behaviour is intentional: a broken satellite file should not stop the
/// app from loading the rest of the fleet.
///
/// # Errors
///
/// Returns [`ConfigError::Io`] only if the directory itself cannot be
/// listed. Per-file failures are logged, not returned.
pub fn load_all_satellites(dir: &Path) -> Result<Vec<SatelliteDefinition>, ConfigError> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("warning: skipping unreadable directory entry: {e}");
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        match load_satellite(&path) {
            Ok(def) => out.push(def),
            Err(e) => {
                eprintln!("warning: skipping {}: {e}", path.display());
            }
        }
    }
    Ok(out)
}

fn validate(def: &SatelliteDefinition) -> Result<(), ConfigError> {
    for d in &def.downlinks {
        if d.baudrate == 0 && !matches!(d.image, Some(ImageDef::Sstv {})) {
            return Err(ConfigError::InvalidValue {
                field: format!("downlink[{}].baudrate", d.label),
                reason: "must be > 0".into(),
            });
        }
        if let Some(schema_name) = &d.telemetry_schema {
            if !def.telemetry.contains_key(schema_name) {
                return Err(ConfigError::InvalidValue {
                    field: format!("downlink[{}].telemetry_schema", d.label),
                    reason: format!("references unknown schema '{schema_name}'"),
                });
            }
        }
        validate_downlink_pipeline(d)?;
    }
    for (name, schema) in &def.telemetry {
        validate_telemetry_schema(name, schema)?;
    }
    Ok(())
}

fn validate_telemetry_schema(name: &str, schema: &TelemetrySchemaDef) -> Result<(), ConfigError> {
    if let Some(disc) = &schema.discriminator {
        if !matches!(disc.length, 1 | 2 | 4 | 8) {
            return Err(ConfigError::InvalidValue {
                field: format!("telemetry.{name}.discriminator.length"),
                reason: "must be 1, 2, 4, or 8".into(),
            });
        }
    }
    for (idx, len) in schema.match_lengths.iter().enumerate() {
        if *len == 0 {
            return Err(ConfigError::InvalidValue {
                field: format!("telemetry.{name}.match_lengths[{idx}]"),
                reason: "must be > 0".into(),
            });
        }
    }
    if !schema.variants.is_empty() && schema.discriminator.is_none() {
        return Err(ConfigError::InvalidValue {
            field: format!("telemetry.{name}"),
            reason: "variants require a discriminator".into(),
        });
    }
    let mut seen = std::collections::HashSet::new();
    for variant in &schema.variants {
        let candidates: Vec<u64> = if variant.match_values.is_empty() {
            vec![variant.match_value]
        } else {
            variant.match_values.clone()
        };
        for value in candidates {
            if !seen.insert(value) {
                return Err(ConfigError::InvalidValue {
                    field: format!("telemetry.{name}.variant[{}]", variant.name),
                    reason: format!("duplicate discriminator value {value}"),
                });
            }
        }
        validate_telemetry_fields(
            &format!("telemetry.{name}.variant[{}]", variant.name),
            &variant.fields,
        )?;
    }
    validate_telemetry_fields(&format!("telemetry.{name}"), &schema.fields)
}

fn validate_telemetry_fields(scope: &str, fields: &[TelemetryFieldDef]) -> Result<(), ConfigError> {
    for field in fields {
        let has_byte = field.offset.is_some() || field.length.is_some();
        let has_bit = field.bit_offset.is_some() || field.bit_length.is_some();
        if has_byte && has_bit {
            return Err(ConfigError::InvalidValue {
                field: format!("{scope}.field[{}]", field.name),
                reason: "must use either byte-level (offset/length) or bit-level (bit_offset/bit_length) addressing, not both".into(),
            });
        }
        if has_byte {
            if field.offset.is_none() || field.length.is_none() {
                return Err(ConfigError::InvalidValue {
                    field: format!("{scope}.field[{}]", field.name),
                    reason: "byte-level addressing requires both offset and length".into(),
                });
            }
            if let Some(length) = field.length {
                if length == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: format!("{scope}.field[{}].length", field.name),
                        reason: "must be > 0".into(),
                    });
                }
                let required = match field.encoding {
                    FieldEncoding::Float32 => Some(4),
                    FieldEncoding::Float64 => Some(8),
                    FieldEncoding::BcdSplit => Some(2),
                    FieldEncoding::SignMagnitude8 => Some(1),
                    FieldEncoding::Q15Signed => Some(2),
                    FieldEncoding::Integer => None,
                };
                if let Some(req) = required {
                    if length != req {
                        return Err(ConfigError::InvalidValue {
                            field: format!("{scope}.field[{}].length", field.name),
                            reason: format!(
                                "encoding {:?} requires length = {req}",
                                field.encoding
                            ),
                        });
                    }
                }
            }
        } else if has_bit {
            if field.bit_offset.is_none() || field.bit_length.is_none() {
                return Err(ConfigError::InvalidValue {
                    field: format!("{scope}.field[{}]", field.name),
                    reason: "bit-level addressing requires both bit_offset and bit_length".into(),
                });
            }
            if let Some(length) = field.bit_length {
                if length == 0 || length > 64 {
                    return Err(ConfigError::InvalidValue {
                        field: format!("{scope}.field[{}].bit_length", field.name),
                        reason: "must be in 1..=64".into(),
                    });
                }
            }
            if !matches!(field.encoding, FieldEncoding::Integer) {
                return Err(ConfigError::InvalidValue {
                    field: format!("{scope}.field[{}].encoding", field.name),
                    reason: "bit-level fields only support the integer encoding".into(),
                });
            }
        } else {
            return Err(ConfigError::InvalidValue {
                field: format!("{scope}.field[{}]", field.name),
                reason: "must specify either byte-level or bit-level addressing".into(),
            });
        }
        if field.encoding == FieldEncoding::BcdSplit
            && (field.decimal_places == 0 || field.decimal_places > 9)
        {
            return Err(ConfigError::InvalidValue {
                field: format!("{scope}.field[{}].decimal_places", field.name),
                reason: "must be in 1..=9 for bcd_split".into(),
            });
        }
    }
    Ok(())
}

fn validate_downlink_pipeline(d: &DownlinkDef) -> Result<(), ConfigError> {
    if let Some(modem) = &d.modem {
        validate_modem(&d.label, modem)?;
    }
    if let Some(framer) = &d.framer {
        validate_framer(&d.label, framer)?;
    }
    if let Some(fec) = &d.fec {
        validate_fec(&d.label, fec)?;
    }
    Ok(())
}

fn validate_modem(label: &str, modem: &ModemDef) -> Result<(), ConfigError> {
    match modem {
        ModemDef::Afsk { mark_hz, space_hz } => {
            if *mark_hz <= 0.0 {
                return invalid_value(format!("downlink[{label}].modem.mark_hz"), "must be > 0");
            }
            if *space_hz <= 0.0 {
                return invalid_value(format!("downlink[{label}].modem.space_hz"), "must be > 0");
            }
        }
        ModemDef::Cpm {
            modulation_index,
            frequency_offset_hz,
            gaussian_bt,
            ..
        } => {
            if modulation_index.is_some_and(|value| value <= 0.0) {
                return invalid_value(
                    format!("downlink[{label}].modem.modulation_index"),
                    "must be > 0",
                );
            }
            if !frequency_offset_hz.is_finite() {
                return invalid_value(
                    format!("downlink[{label}].modem.frequency_offset_hz"),
                    "must be finite",
                );
            }
            if gaussian_bt.is_some_and(|value| value <= 0.0) {
                return invalid_value(
                    format!("downlink[{label}].modem.gaussian_bt"),
                    "must be > 0",
                );
            }
        }
        ModemDef::Linear {
            frequency_offset_hz,
            matched_filter_rolloff,
            ..
        } => {
            if !frequency_offset_hz.is_finite() {
                return invalid_value(
                    format!("downlink[{label}].modem.frequency_offset_hz"),
                    "must be finite",
                );
            }
            if !matched_filter_rolloff.is_finite()
                || *matched_filter_rolloff < 0.0
                || *matched_filter_rolloff > 1.0
            {
                return invalid_value(
                    format!("downlink[{label}].modem.matched_filter_rolloff"),
                    "must be in [0.0, 1.0]",
                );
            }
        }
        ModemDef::Lora {
            spreading_factor,
            bandwidth_hz,
        } => {
            if !(6..=12).contains(spreading_factor) {
                return invalid_value(
                    format!("downlink[{label}].modem.spreading_factor"),
                    "must be between 6 and 12",
                );
            }
            if *bandwidth_hz == 0 {
                return invalid_value(
                    format!("downlink[{label}].modem.bandwidth_hz"),
                    "must be > 0",
                );
            }
        }
        ModemDef::FourFsk { freq_offsets_hz } => {
            if !freq_offsets_hz.iter().all(|value| value.is_finite()) {
                return invalid_value(
                    format!("downlink[{label}].modem.freq_offsets_hz"),
                    "all offsets must be finite",
                );
            }
        }
    }

    Ok(())
}

fn validate_framer(label: &str, framer: &FramerDef) -> Result<(), ConfigError> {
    match framer {
        FramerDef::Hdlc => {}
        FramerDef::Ao40 { threshold } => {
            if *threshold > 65 {
                return invalid_value(
                    format!("downlink[{label}].framer.threshold"),
                    "must be <= 65",
                );
            }
        }
        FramerDef::Syncword {
            syncword,
            threshold,
            payload_bits,
        } => {
            if syncword.is_empty() || !syncword.bytes().all(|byte| matches!(byte, b'0' | b'1')) {
                return invalid_value(
                    format!("downlink[{label}].framer.syncword"),
                    "must contain only 0 and 1 characters",
                );
            }
            if *threshold > syncword.len() {
                return invalid_value(
                    format!("downlink[{label}].framer.threshold"),
                    "must be <= syncword length",
                );
            }
            if *payload_bits == 0 {
                return invalid_value(
                    format!("downlink[{label}].framer.payload_bits"),
                    "must be > 0",
                );
            }
        }
        FramerDef::Ax100Asm { threshold } => {
            if *threshold > 32 {
                return invalid_value(
                    format!("downlink[{label}].framer.threshold"),
                    "must be <= 32",
                );
            }
        }
    }

    Ok(())
}

fn validate_fec(label: &str, fec: &FecDef) -> Result<(), ConfigError> {
    match fec {
        FecDef::ReedSolomon { interleave } => {
            if *interleave == 0 {
                return invalid_value(format!("downlink[{label}].fec.interleave"), "must be > 0");
            }
        }
        FecDef::Ao40 | FecDef::Ax100 => {}
    }

    Ok(())
}

fn invalid_value<T>(field: String, reason: &'static str) -> Result<T, ConfigError> {
    Err(ConfigError::InvalidValue {
        field,
        reason: reason.to_string(),
    })
}

fn translate_toml_error(err: toml::de::Error) -> ConfigError {
    let msg = err.message();
    if let Some(field) = missing_field_name(msg) {
        return ConfigError::MissingField(field);
    }
    ConfigError::Toml(err)
}

fn missing_field_name(msg: &str) -> Option<String> {
    let marker = "missing field `";
    let start = msg.find(marker)? + marker.len();
    let rest = &msg[start..];
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("openhoshimi-test-{}", std::process::id()));
        if let Err(err) = std::fs::create_dir_all(&dir) {
            panic!("create temp dir: {err}");
        }
        let path = dir.join(name);
        let mut f = match std::fs::File::create(&path) {
            Ok(file) => file,
            Err(err) => panic!("create file: {err}"),
        };
        if let Err(err) = f.write_all(contents.as_bytes()) {
            panic!("write file: {err}");
        }
        path
    }

    const VALID_TOML: &str = r#"
[satellite]
name = "AO-73"
aliases = ["FUNcube-1"]
norad_id = 39444

[[downlink]]
label = "1k2 BPSK beacon"
freq_hz = 145935000
modulation = "DBPSK"
baudrate = 1200
framing = "AO40_FEC"
telemetry_schema = "funcube_1"

[downlink.modem]
kind = "linear"
mode = "dbpsk"
frequency_offset_hz = -6000.0
differential = true
swap_iq = true

[downlink.framer]
kind = "ao40"
threshold = 0

[downlink.fec]
kind = "ao40"

[downlink.codec]
kind = "ao40_fec"

[telemetry.funcube_1]

[[telemetry.funcube_1.field]]
name = "bat_voltage"
group = "eps"
offset = 2
length = 2
endian = "big"
scale = 0.001
unit = "V"
warn_below = 6.5
warn_above = 8.4
"#;

    #[test]
    fn load_valid_toml() {
        let path = write_temp("valid.toml", VALID_TOML);
        let def = match load_satellite(&path) {
            Ok(def) => def,
            Err(err) => panic!("valid toml should load: {err}"),
        };
        assert_eq!(def.satellite.norad_id, 39444);
        assert_eq!(def.downlinks.len(), 1);
        assert_eq!(def.downlinks[0].baudrate, 1200);
        match &def.downlinks[0].modem {
            Some(ModemDef::Linear {
                mode,
                frequency_offset_hz,
                differential,
                invert,
                swap_iq,
                carrier_loop_bandwidth,
                frequency_loop_bandwidth,
                nco_max_offset_hz,
                matched_filter_rolloff,
                matched_filter_span_symbols,
            }) => {
                assert_eq!(*mode, LinearModeDef::Dbpsk);
                assert_eq!(*frequency_offset_hz, -6000.0);
                assert!(*differential);
                assert!(!*invert);
                assert!(*swap_iq);
                assert_eq!(*carrier_loop_bandwidth, 0.0);
                assert_eq!(*frequency_loop_bandwidth, 0.0);
                assert_eq!(*nco_max_offset_hz, 0.0);
                assert_eq!(*matched_filter_rolloff, 0.0);
                assert_eq!(*matched_filter_span_symbols, 0);
            }
            other => panic!("expected linear modem, got {other:?}"),
        }
        match &def.downlinks[0].framer {
            Some(FramerDef::Ao40 { threshold }) => assert_eq!(*threshold, 0),
            other => panic!("expected AO-40 framer, got {other:?}"),
        }
        assert!(matches!(def.downlinks[0].fec, Some(FecDef::Ao40)));
        assert!(matches!(def.downlinks[0].codec, Some(CodecDef::Ao40Fec)));
        let schema = match def.telemetry.get("funcube_1") {
            Some(schema) => schema,
            None => panic!("schema present"),
        };
        assert_eq!(schema.fields.len(), 1);
        assert_eq!(schema.fields[0].name, "bat_voltage");
        assert_eq!(schema.fields[0].endian, Endian::Big);
    }

    #[test]
    fn missing_required_field_is_reported() {
        // norad_id is required and is missing here.
        let bad = r#"
[satellite]
name = "X"

[[downlink]]
label = "x"
freq_hz = 1
modulation = "FSK"
baudrate = 1200
framing = "AX25"
"#;
        let path = write_temp("missing.toml", bad);
        let err = match load_satellite(&path) {
            Ok(_) => panic!("should fail"),
            Err(err) => err,
        };
        match err {
            ConfigError::MissingField(name) => assert_eq!(name, "norad_id"),
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    #[test]
    fn invalid_syncword_is_reported() {
        let bad = r#"
[satellite]
name = "X"
norad_id = 1

[[downlink]]
label = "x"
freq_hz = 1
modulation = "FSK"
baudrate = 1200
framing = "UNKNOWN"

[downlink.framer]
kind = "syncword"
syncword = "10x1"
payload_bits = 8
"#;
        let path = write_temp("invalid-syncword.toml", bad);
        let err = match load_satellite(&path) {
            Ok(_) => panic!("should fail"),
            Err(err) => err,
        };

        match err {
            ConfigError::InvalidValue { field, reason } => {
                assert_eq!(field, "downlink[x].framer.syncword");
                assert_eq!(reason, "must contain only 0 and 1 characters");
            }
            other => panic!("expected InvalidValue, got {other:?}"),
        }
    }

    #[test]
    fn load_all_skips_broken_files() {
        let dir =
            std::env::temp_dir().join(format!("openhoshimi-test-loadall-{}", std::process::id()));
        // Clean any previous run.
        let _ = std::fs::remove_dir_all(&dir);
        if let Err(err) = std::fs::create_dir_all(&dir) {
            panic!("create dir: {err}");
        }

        let good = dir.join("good.toml");
        if let Err(err) = std::fs::write(&good, VALID_TOML) {
            panic!("write good: {err}");
        }
        let bad = dir.join("bad.toml");
        if let Err(err) = std::fs::write(&bad, "this is not = valid [toml") {
            panic!("write bad: {err}");
        }
        let ignored = dir.join("notes.txt");
        if let Err(err) = std::fs::write(&ignored, "ignore me") {
            panic!("write ignored: {err}");
        }

        let defs = match load_all_satellites(&dir) {
            Ok(defs) => defs,
            Err(err) => panic!("directory readable: {err}"),
        };
        assert_eq!(defs.len(), 1, "only the good file should load");
        assert_eq!(defs[0].satellite.name, "AO-73");
    }

    #[test]
    fn sstv_downlink_loads_without_pipeline_stages() {
        // SSTV downlinks have no modem/framer/codec and a meaningless
        // baudrate; only the [downlink.image] block with protocol "sstv"
        // is mandatory. The validator must accept this shape.
        let toml = r#"
[satellite]
name = "ISS"
norad_id = 25544

[[downlink]]
label = "SSTV image"
freq_hz = 145800000
modulation = "FM"
baudrate = 0
framing = "none"

[downlink.image]
protocol = "sstv"
"#;
        let path = write_temp("sstv.toml", toml);
        let def = match load_satellite(&path) {
            Ok(def) => def,
            Err(err) => panic!("sstv toml should load: {err}"),
        };
        assert_eq!(def.downlinks.len(), 1);
        assert!(matches!(def.downlinks[0].image, Some(ImageDef::Sstv {})));
        assert!(def.downlinks[0].modem.is_none());
        assert!(def.downlinks[0].framer.is_none());
        assert!(def.downlinks[0].codec.is_none());
    }
}
