//! Image-frame reassembly.
//!
//! Some satellites send images split across many small frames. Each
//! frame carries a fixed-size raw chunk and a byte offset into the
//! linear stream that the chunks rebuild. Frames arrive out of order
//! and with gaps from the LEO pass edges; the reassembler's job is to
//! slot each chunk into the correct stream position and keep a
//! per-chunk receipt bitmap so the GUI can show what is missing.
//!
//! ## Image boundaries
//!
//! The Geoscan custom protocol (STRATOSAT-TK-1, Geoscan-Edelveis) does
//! not carry an explicit image-id field: the only group byte that
//! looked like an id (the `0x06` at byte 7) is actually a fixed marker
//! shared by every image-bearing frame in a recording. New images are
//! therefore detected at run time, not from a TOML field, by watching
//! for the offset returning to zero with the chunk starting in the
//! decoder-specific start-of-image marker:
//!
//! * `Jpeg` decoder: the JPEG SOI (`FF D8`) at the start of an
//!   offset-0 chunk.
//! * `Raw` decoder: any offset-0 chunk arriving after the current
//!   image already had data written.
//!
//! ## Schema
//!
//! Driven by the satellite TOML's `[downlink.image]` block. v1 ships a
//! single [`GeoscanImageReassembler`] for [`ImageDef::Geoscan`].
//! Adding SSDV later is a new variant of [`ImageDef`] plus a new impl
//! of [`ImageReassembler`]; nothing in the runtime or GUI needs to know
//! the protocol name.

use openhoshimi_core::satellite::{
    ByteOrderDef, ImageDecoderDef, ImageDef, ImageField, PixelFormatDef,
};

use crate::ssdv::{SsdvDecoder, SsdvPacket, SsdvPacketKind};
use crate::ssdv_jpeg::JpegBuilder;

/// Stateful reassembler that consumes raw frame payloads and accumulates
/// one or more image streams over time.
pub trait ImageReassembler: Send {
    /// Feed one decoded frame payload to the reassembler.
    ///
    /// Returns `Some(update)` when the payload was an image chunk and
    /// at least one byte was written into a stream, even if the chunk
    /// is a duplicate of one already received. Returns `None` when the
    /// payload's header does not match the configured signature or the
    /// chunk is structurally invalid (out-of-range offsets etc.).
    fn ingest(&mut self, payload: &[u8]) -> Option<PacketUpdate>;

    /// Produce a snapshot of every reassembled image. Cheap to call on
    /// every UI repaint: images are stored in arrival order and cloned
    /// by the snapshot.
    fn snapshot(&self) -> ImageSnapshot;

    /// Drop all accumulated state and start over. Called from the GUI
    /// when the user clicks "Reset" or switches to a different
    /// downlink / satellite.
    fn reset(&mut self);
}

/// Result of a successful chunk write.
#[derive(Debug, Clone, Copy)]
pub struct PacketUpdate {
    /// Internal image index this chunk belongs to (0 = first image
    /// observed in this session, 1 = next, ...).
    pub image_idx: u32,
    /// Byte offset into the image stream where the chunk was written.
    pub offset: u32,
    /// Number of bytes written.
    pub bytes_written: usize,
    /// Index of the chunk slot in the receipt bitmap.
    pub chunk_idx: u32,
    /// `true` if this was the first time this chunk slot was filled;
    /// `false` if a chunk already existed at the same slot and was
    /// overwritten.
    pub fresh: bool,
    /// `true` if this chunk started a new image (offset 0 + decoder's
    /// start-of-image marker).
    pub started_new_image: bool,
}

/// Read-only view of every reassembled image, suitable for rendering.
#[derive(Debug, Clone)]
pub struct ImageSnapshot {
    /// Decoder used to interpret each image's byte stream.
    pub decoder: ImageDecoder,
    /// Raw-mode canvas width in pixels (ignored for `Jpeg`).
    pub width: u32,
    /// Raw-mode canvas height in pixels (ignored for `Jpeg`).
    pub height: u32,
    /// Raw-mode pixel format (ignored for `Jpeg`).
    pub pixel_format: PixelFormat,
    /// Bytes per chunk slot, mirrored from the TOML's `chunk_bytes`.
    /// Used by the GUI's partial-JPEG preview to truncate the byte
    /// stream at the first missing-chunk boundary.
    pub chunk_bytes: u32,
    /// One entry per image observed in this session, in arrival order.
    pub images: Vec<ImageStream>,
}

/// Reassembled state for a single image.
#[derive(Debug, Clone)]
pub struct ImageStream {
    /// Internal index assigned in arrival order.
    pub image_idx: u32,
    /// Reassembled byte stream. Length grows on demand: each new chunk
    /// extends the buffer to cover at least `offset + chunk.len()`.
    pub bytes: Vec<u8>,
    /// One bit per chunk slot covered by `bytes`; `true` when a chunk
    /// has been received for that slot.
    pub received_chunks: Vec<u64>,
    /// Number of chunk slots covered by `bytes` (rounded up).
    pub total_chunks: u32,
    /// Number of unique chunk slots that have received at least one
    /// chunk. Capped at `total_chunks`.
    pub received_count: u32,
    /// Byte offset of the most recently written chunk.
    pub last_offset: u32,
    /// Slot index of the most recently written chunk.
    pub last_chunk_idx: u32,
}

impl ImageStream {
    /// Test whether the chunk slot at `idx` has ever been written.
    pub fn chunk_received(&self, idx: u32) -> bool {
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        self.received_chunks
            .get(word)
            .is_some_and(|w| (w >> bit) & 1 == 1)
    }
}

/// Decoder mirror, decoupled from the TOML enum so callers don't have
/// to depend on `openhoshimi_core::satellite` to know the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageDecoder {
    /// Reassembled bytes are a JPEG bitstream.
    Jpeg,
    /// Reassembled bytes are raw pixels.
    Raw,
}

impl From<ImageDecoderDef> for ImageDecoder {
    fn from(value: ImageDecoderDef) -> Self {
        match value {
            ImageDecoderDef::Jpeg => ImageDecoder::Jpeg,
            ImageDecoderDef::Raw => ImageDecoder::Raw,
        }
    }
}

/// Pixel format mirror for `Raw` decoder mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit grayscale; one byte per pixel.
    Gray8,
    /// 16-bit RGB565; two bytes per pixel, big-endian.
    Rgb565,
    /// 24-bit RGB888; three bytes per pixel.
    Rgb888,
}

impl PixelFormat {
    /// Bytes occupied by one pixel.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            PixelFormat::Gray8 => 1,
            PixelFormat::Rgb565 => 2,
            PixelFormat::Rgb888 => 3,
        }
    }
}

impl From<PixelFormatDef> for PixelFormat {
    fn from(value: PixelFormatDef) -> Self {
        match value {
            PixelFormatDef::Gray8 => PixelFormat::Gray8,
            PixelFormatDef::Rgb565 => PixelFormat::Rgb565,
            PixelFormatDef::Rgb888 => PixelFormat::Rgb888,
        }
    }
}

/// Build a reassembler from a satellite TOML definition.
///
/// Validates the TOML's hex `header_signature`, byte-offset ranges, and
/// chunk geometry so a misconfigured downlink is rejected up-front
/// instead of producing garbage at runtime.
///
/// # Errors
///
/// Returns a human-readable `String` if the TOML is structurally
/// invalid (bad hex, fields overlapping the chunk, zero-area canvas,
/// etc.).
pub fn build(def: &ImageDef) -> Result<Box<dyn ImageReassembler>, String> {
    match def {
        ImageDef::Geoscan {
            header_signature,
            offset_field,
            chunk_at,
            chunk_bytes,
            decoder,
            width,
            height,
            pixel_format,
        } => {
            let header =
                parse_hex(header_signature).map_err(|e| format!("image.header_signature: {e}"))?;
            validate_field("offset_field", offset_field, 1..=4)?;
            if *chunk_bytes == 0 {
                return Err("image.chunk_bytes must be > 0".to_string());
            }
            let decoder: ImageDecoder = (*decoder).into();
            if decoder == ImageDecoder::Raw && (*width == 0 || *height == 0) {
                return Err(
                    "image.width / image.height must be > 0 when decoder = \"raw\"".to_string(),
                );
            }
            Ok(Box::new(GeoscanImageReassembler {
                header,
                offset_field: offset_field.clone(),
                chunk_at: *chunk_at,
                chunk_bytes: *chunk_bytes,
                decoder,
                width: *width,
                height: *height,
                pixel_format: PixelFormat::from(*pixel_format),
                images: Vec::new(),
                next_image_idx: 0,
            }))
        }
        ImageDef::Ssdv { callsign } => Ok(Box::new(SsdvImageReassembler::new(callsign.clone()))),
        ImageDef::Sstv {} => Err(
            "image.protocol = \"sstv\" is decoded by SstvAnalyzer, not by build_image_reassembler"
                .to_string(),
        ),
    }
}

fn validate_field(
    name: &str,
    field: &ImageField,
    allowed_lens: std::ops::RangeInclusive<usize>,
) -> Result<(), String> {
    if field.len == 0 {
        return Err(format!("image.{name}.len must be > 0"));
    }
    if !allowed_lens.contains(&field.len) {
        return Err(format!(
            "image.{name}.len = {} out of range {:?}",
            field.len, allowed_lens
        ));
    }
    Ok(())
}

fn parse_hex(text: &str) -> Result<Vec<u8>, String> {
    let cleaned: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.is_empty() {
        return Err("hex string is empty".to_string());
    }
    if cleaned.len() % 2 != 0 {
        return Err(format!(
            "hex string must have an even nibble count, got {}",
            cleaned.len()
        ));
    }
    let mut out = Vec::with_capacity(cleaned.len() / 2);
    for i in (0..cleaned.len()).step_by(2) {
        let byte = u8::from_str_radix(&cleaned[i..i + 2], 16)
            .map_err(|e| format!("bad hex at offset {}: {}", i, e))?;
        out.push(byte);
    }
    Ok(out)
}

fn read_uint(bytes: &[u8], field: &ImageField) -> Option<u32> {
    let slice = bytes.get(field.at..field.at.checked_add(field.len)?)?;
    let mut value: u32 = 0;
    match field.endian {
        ByteOrderDef::Be => {
            for &b in slice {
                value = (value << 8) | b as u32;
            }
        }
        ByteOrderDef::Le => {
            for (i, &b) in slice.iter().enumerate() {
                value |= (b as u32) << (8 * i);
            }
        }
    }
    Some(value)
}

/// Geoscan-protocol reassembler. Frames whose first
/// `header_signature.len()` bytes match the configured signature are
/// treated as image chunks; all other payloads are ignored.
pub struct GeoscanImageReassembler {
    header: Vec<u8>,
    offset_field: ImageField,
    chunk_at: usize,
    chunk_bytes: usize,
    decoder: ImageDecoder,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    images: Vec<GeoscanImageState>,
    next_image_idx: u32,
}

struct GeoscanImageState {
    image_idx: u32,
    bytes: Vec<u8>,
    received: Vec<u64>,
    total_chunks: u32,
    received_count: u32,
    last_offset: u32,
    last_chunk_idx: u32,
}

impl GeoscanImageState {
    fn new(image_idx: u32) -> Self {
        Self {
            image_idx,
            bytes: Vec::new(),
            received: Vec::new(),
            total_chunks: 0,
            received_count: 0,
            last_offset: 0,
            last_chunk_idx: 0,
        }
    }

    fn ensure_capacity(&mut self, end: usize, chunk_bytes: usize) {
        if end > self.bytes.len() {
            self.bytes.resize(end, 0);
        }
        let needed_chunks = end.div_ceil(chunk_bytes) as u32;
        if needed_chunks > self.total_chunks {
            self.total_chunks = needed_chunks;
            let needed_words = (self.total_chunks as usize).div_ceil(64);
            if needed_words > self.received.len() {
                self.received.resize(needed_words, 0);
            }
        }
    }

    fn mark(&mut self, idx: u32) -> bool {
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        let mask = 1u64 << bit;
        if self.received[word] & mask == 0 {
            self.received[word] |= mask;
            self.received_count += 1;
            true
        } else {
            false
        }
    }

    fn snapshot(&self) -> ImageStream {
        ImageStream {
            image_idx: self.image_idx,
            bytes: self.bytes.clone(),
            received_chunks: self.received.clone(),
            total_chunks: self.total_chunks,
            received_count: self.received_count,
            last_offset: self.last_offset,
            last_chunk_idx: self.last_chunk_idx,
        }
    }
}

impl GeoscanImageReassembler {
    /// Decide whether the chunk arriving at offset 0 starts a new
    /// image. The current image stays open until a chunk shows up that
    /// (a) lands at offset 0, and (b) actually carries a fresh stream
    /// according to the configured decoder. Without (b) a duplicate
    /// retransmission of the very first chunk would orphan the rest of
    /// the previous image.
    /// Decide whether a chunk arriving at offset 0 should split: it
    /// starts a new image when the previous image already has data
    /// AND, for JPEG, the chunk begins with the SOI marker (FF D8).
    /// Without the SOI guard, a retransmission of the very first
    /// chunk would orphan the rest of the previous image. Returns
    /// `true` for the very first SOI of a session (no images yet) so
    /// the caller can use the same predicate to gate JPEG-mode start.
    fn is_new_image_start(&self, chunk: &[u8]) -> bool {
        match self.decoder {
            ImageDecoder::Jpeg => chunk.len() >= 2 && chunk[0] == 0xFF && chunk[1] == 0xD8,
            ImageDecoder::Raw => self
                .images
                .last()
                .is_some_and(|i| i.received_count > 0 || !i.bytes.is_empty()),
        }
    }
}

impl ImageReassembler for GeoscanImageReassembler {
    fn ingest(&mut self, payload: &[u8]) -> Option<PacketUpdate> {
        if payload.len() < self.header.len() {
            return None;
        }
        if payload[..self.header.len()] != self.header[..] {
            return None;
        }
        let offset = read_uint(payload, &self.offset_field)?;
        let end_in_payload = self.chunk_at.checked_add(self.chunk_bytes)?;
        if end_in_payload > payload.len() {
            return None;
        }
        let chunk = &payload[self.chunk_at..end_in_payload];
        let chunk_idx = offset / self.chunk_bytes as u32;

        let mut started_new_image = false;
        if offset == 0 && self.is_new_image_start(chunk) {
            started_new_image = true;
        }
        // Drop chunks received before any image has started in JPEG
        // mode: without an SOI we have no anchor for the bytestream,
        // and the decoder will reject whatever we emit. Raw mode does
        // not need an anchor (every byte slots into a fixed-size frame
        // buffer), so we keep the legacy behaviour there.
        if self.images.is_empty()
            && !started_new_image
            && matches!(self.decoder, ImageDecoder::Jpeg)
        {
            return None;
        }
        if self.images.is_empty() || started_new_image {
            let idx = self.next_image_idx;
            self.next_image_idx += 1;
            self.images.push(GeoscanImageState::new(idx));
        }

        let last_idx = self.images.len() - 1;
        let state = &mut self.images[last_idx];
        let end = (offset as usize).checked_add(chunk.len())?;
        state.ensure_capacity(end, self.chunk_bytes);
        state.bytes[offset as usize..end].copy_from_slice(chunk);
        let fresh = state.mark(chunk_idx);
        state.last_offset = offset;
        state.last_chunk_idx = chunk_idx;

        Some(PacketUpdate {
            image_idx: state.image_idx,
            offset,
            bytes_written: chunk.len(),
            chunk_idx,
            fresh,
            started_new_image,
        })
    }

    fn snapshot(&self) -> ImageSnapshot {
        ImageSnapshot {
            decoder: self.decoder,
            width: self.width,
            height: self.height,
            pixel_format: self.pixel_format,
            chunk_bytes: self.chunk_bytes as u32,
            images: self
                .images
                .iter()
                .map(GeoscanImageState::snapshot)
                .collect(),
        }
    }

    fn reset(&mut self) {
        self.images.clear();
        self.next_image_idx = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def_jpeg() -> ImageDef {
        ImageDef::Geoscan {
            header_signature: "02 00 3E 05 09".to_string(),
            offset_field: ImageField {
                at: 5,
                len: 2,
                endian: ByteOrderDef::Le,
            },
            chunk_at: 8,
            chunk_bytes: 56,
            decoder: ImageDecoderDef::Jpeg,
            width: 320,
            height: 240,
            pixel_format: PixelFormatDef::Gray8,
        }
    }

    fn def_raw() -> ImageDef {
        ImageDef::Geoscan {
            header_signature: "AA".to_string(),
            offset_field: ImageField {
                at: 1,
                len: 2,
                endian: ByteOrderDef::Be,
            },
            chunk_at: 3,
            chunk_bytes: 8,
            decoder: ImageDecoderDef::Raw,
            width: 8,
            height: 8,
            pixel_format: PixelFormatDef::Gray8,
        }
    }

    fn frame_jpeg(offset: u16, first_two: [u8; 2], fill: u8) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        buf[0..5].copy_from_slice(&[0x02, 0x00, 0x3E, 0x05, 0x09]);
        buf[5..7].copy_from_slice(&offset.to_le_bytes());
        buf[7] = 0x06;
        buf[8] = first_two[0];
        buf[9] = first_two[1];
        for b in &mut buf[10..64] {
            *b = fill;
        }
        buf
    }

    #[test]
    fn ingest_writes_chunk_at_offset() {
        let mut r = build(&def_jpeg()).unwrap();
        let update = r.ingest(&frame_jpeg(0, [0xFF, 0xD8], 0xAA)).unwrap();
        assert_eq!(update.image_idx, 0);
        assert_eq!(update.offset, 0);
        assert_eq!(update.bytes_written, 56);
        assert!(update.fresh);
        // The very first SOI chunk also "starts a new image" under the
        // JPEG-mode rule that requires every JPEG stream to begin with
        // FF D8 — there's no prior image, but this is the anchor.
        assert!(update.started_new_image);
        let snap = r.snapshot();
        assert_eq!(snap.images.len(), 1);
        assert_eq!(snap.images[0].received_count, 1);
        assert_eq!(snap.images[0].bytes[..2], [0xFF, 0xD8]);
    }

    #[test]
    fn header_mismatch_returns_none() {
        let mut r = build(&def_jpeg()).unwrap();
        let mut bad = frame_jpeg(0, [0xFF, 0xD8], 0xAA);
        bad[0] = 0xFF;
        assert!(r.ingest(&bad).is_none());
    }

    #[test]
    fn out_of_order_arrival_grows_buffer() {
        let mut r = build(&def_jpeg()).unwrap();
        // First chunk arrives at offset 0 with the JPEG SOI; subsequent
        // chunks arrive out of order at higher offsets. They must all
        // accumulate in the same image because none of them is a new
        // SOI at offset 0.
        let _ = r.ingest(&frame_jpeg(0, [0xFF, 0xD8], 0xAA)).unwrap();
        let _ = r.ingest(&frame_jpeg(112, [0xCC, 0xCC], 0xCC)).unwrap();
        let _ = r.ingest(&frame_jpeg(56, [0xBB, 0xBB], 0xBB)).unwrap();
        let snap = r.snapshot();
        assert_eq!(snap.images.len(), 1);
        assert_eq!(snap.images[0].received_count, 3);
        assert_eq!(snap.images[0].bytes[0], 0xFF);
        assert_eq!(snap.images[0].bytes[56], 0xBB);
        assert_eq!(snap.images[0].bytes[112], 0xCC);
    }

    #[test]
    fn jpeg_soi_at_offset_zero_starts_new_image() {
        let mut r = build(&def_jpeg()).unwrap();
        let _ = r.ingest(&frame_jpeg(0, [0xFF, 0xD8], 0xAA)).unwrap();
        let _ = r.ingest(&frame_jpeg(56, [0xBB, 0xBB], 0xBB)).unwrap();
        let update = r.ingest(&frame_jpeg(0, [0xFF, 0xD8], 0xCC)).unwrap();
        assert!(update.started_new_image);
        assert_eq!(update.image_idx, 1);
        let snap = r.snapshot();
        assert_eq!(snap.images.len(), 2);
        assert_eq!(snap.images[0].received_count, 2);
        assert_eq!(snap.images[1].received_count, 1);
    }

    #[test]
    fn duplicate_offset_zero_without_soi_does_not_split() {
        let mut r = build(&def_jpeg()).unwrap();
        let _ = r.ingest(&frame_jpeg(0, [0xFF, 0xD8], 0xAA)).unwrap();
        let _ = r.ingest(&frame_jpeg(56, [0xBB, 0xBB], 0xBB)).unwrap();
        // A retransmission of offset 0 that does NOT start with FF D8
        // (e.g. wrong-PN9 corrupted payload) must not split images.
        let update = r.ingest(&frame_jpeg(0, [0x00, 0x00], 0xAB)).unwrap();
        assert!(!update.started_new_image);
        assert_eq!(update.image_idx, 0);
        assert_eq!(r.snapshot().images.len(), 1);
    }

    #[test]
    fn raw_decoder_offset_zero_after_data_splits() {
        let mut r = build(&def_raw()).unwrap();
        let mut buf = vec![0xAA, 0x00, 0x00];
        buf.extend_from_slice(&[1u8; 8]);
        let _ = r.ingest(&buf).unwrap();
        // Different chunk at offset 0 -> new image.
        let mut buf2 = vec![0xAA, 0x00, 0x00];
        buf2.extend_from_slice(&[2u8; 8]);
        let update = r.ingest(&buf2).unwrap();
        assert!(update.started_new_image);
        assert_eq!(r.snapshot().images.len(), 2);
    }

    #[test]
    fn reset_clears_images() {
        let mut r = build(&def_jpeg()).unwrap();
        let _ = r.ingest(&frame_jpeg(0, [0xFF, 0xD8], 0xAA)).unwrap();
        r.reset();
        assert!(r.snapshot().images.is_empty());
    }

    #[test]
    fn jpeg_chunks_before_first_soi_are_dropped() {
        // In a real recording, listening picks up mid-image bytes long
        // before the first SOI arrives. Without an SOI anchor a JPEG
        // bytestream has no decoder entry point, so those chunks must
        // be dropped — otherwise they accumulate as a phantom image
        // #0 that decode_file later writes as a useless `.jpg` file.
        let mut r = build(&def_jpeg()).unwrap();
        assert!(r.ingest(&frame_jpeg(56, [0xBB, 0xBB], 0xBB)).is_none());
        assert!(r.ingest(&frame_jpeg(112, [0xCC, 0xCC], 0xCC)).is_none());
        assert!(r.snapshot().images.is_empty());
        // The first SOI then anchors image #0 and subsequent chunks
        // accumulate into it normally.
        let update = r.ingest(&frame_jpeg(0, [0xFF, 0xD8], 0xAA)).unwrap();
        assert!(update.started_new_image);
        assert_eq!(update.image_idx, 0);
        let _ = r.ingest(&frame_jpeg(56, [0xBB, 0xBB], 0xBB)).unwrap();
        assert_eq!(r.snapshot().images.len(), 1);
        assert_eq!(r.snapshot().images[0].received_count, 2);
    }
}

// ============================================================================
// SSDV reassembler
// ============================================================================

/// SSDV-protocol image reassembler.
///
/// Each `ingest` call expects a *full 256-byte SSDV packet* (the
/// reassembler re-runs the decoder rather than trusting upstream
/// header parsing). Packets are bucketed by `image_id`; within each
/// bucket they slot by `packet_id`. The byte stream produced by
/// [`ImageReassembler::snapshot`] is a complete JPEG that any
/// off-the-shelf decoder accepts: SSDV's standard JFIF header,
/// quality-scaled DQT, fixed Huffman tables, SOF0 derived from the
/// packet's `width`/`height`/`mcu_mode`, SOS, then the SSDV bitstream
/// re-coded into proper differential-DC JPEG entropy data with
/// byte-stuffing applied. Missing packets are filled with
/// zero-valued DC + EOB AC codes (no JPEG restart markers), matching
/// fsphil's reference decoder.
///
/// Lifted from the SSDV reference implementation
/// ([fsphil/ssdv](https://github.com/fsphil/ssdv), GPL-3.0-or-later).
pub struct SsdvImageReassembler {
    decoder: SsdvDecoder,
    callsign_filter: Option<String>,
    images: Vec<SsdvImageState>,
    next_image_idx: u32,
}

struct SsdvImageState {
    image_idx: u32,
    image_id: u8,
    width: u16,
    height: u16,
    quality: u8,
    mcu_mode: u8,
    packet_kind: SsdvPacketKind,
    packets: Vec<Option<SsdvPacket>>,
    received: Vec<u64>,
    received_count: u32,
    last_packet_id: u16,
    eoi_seen: bool,
}

impl SsdvImageReassembler {
    /// Construct a reassembler. `callsign_filter`, when set, ignores
    /// packets whose decoded callsign does not match.
    pub fn new(callsign_filter: Option<String>) -> Self {
        Self {
            decoder: SsdvDecoder::new(),
            callsign_filter,
            images: Vec::new(),
            next_image_idx: 0,
        }
    }

    fn image_for(&mut self, packet: &SsdvPacket) -> usize {
        if let Some(idx) = self
            .images
            .iter()
            .position(|state| state.image_id == packet.image_id)
        {
            return idx;
        }
        let idx = self.next_image_idx;
        self.next_image_idx += 1;
        self.images.push(SsdvImageState {
            image_idx: idx,
            image_id: packet.image_id,
            width: packet.width,
            height: packet.height,
            quality: packet.quality,
            mcu_mode: packet.mcu_mode,
            packet_kind: packet.kind,
            packets: Vec::new(),
            received: Vec::new(),
            received_count: 0,
            last_packet_id: 0,
            eoi_seen: false,
        });
        self.images.len() - 1
    }
}

impl ImageReassembler for SsdvImageReassembler {
    fn ingest(&mut self, payload: &[u8]) -> Option<PacketUpdate> {
        let packet = self.decoder.decode(payload).ok()?;
        if let Some(filter) = &self.callsign_filter {
            if !packet.callsign.eq_ignore_ascii_case(filter) {
                return None;
            }
        }
        let was_empty = self
            .images
            .iter()
            .all(|state| state.image_id != packet.image_id);
        let idx = self.image_for(&packet);
        let started_new_image = was_empty;
        let state = &mut self.images[idx];
        // Update image-level metadata each packet — the encoder
        // sends the same width/height/quality/mcu_mode in every
        // packet of a given image, but we re-latch in case an early
        // packet was lost and a later one carries different bytes.
        state.width = packet.width;
        state.height = packet.height;
        state.quality = packet.quality;
        state.mcu_mode = packet.mcu_mode;
        state.packet_kind = packet.kind;
        if packet.eoi {
            state.eoi_seen = true;
        }
        let pkt_id = packet.packet_id as usize;
        if pkt_id >= state.packets.len() {
            state.packets.resize_with(pkt_id + 1, || None);
            let needed_words = (pkt_id + 1).div_ceil(64);
            if needed_words > state.received.len() {
                state.received.resize(needed_words, 0);
            }
        }
        let bit_word = pkt_id / 64;
        let bit_mask = 1u64 << (pkt_id % 64);
        let fresh = state.received[bit_word] & bit_mask == 0;
        if fresh {
            state.received[bit_word] |= bit_mask;
            state.received_count += 1;
        }
        let payload_len = packet.payload.len();
        state.packets[pkt_id] = Some(packet.clone());
        state.last_packet_id = packet.packet_id;
        Some(PacketUpdate {
            image_idx: state.image_idx,
            offset: pkt_id as u32,
            bytes_written: payload_len,
            chunk_idx: pkt_id as u32,
            fresh,
            started_new_image,
        })
    }

    fn snapshot(&self) -> ImageSnapshot {
        let images: Vec<ImageStream> = self
            .images
            .iter()
            .map(|state| {
                let bytes = build_jpeg(state);
                let total_chunks = state.packets.len() as u32;
                ImageStream {
                    image_idx: state.image_idx,
                    bytes,
                    received_chunks: state.received.clone(),
                    total_chunks,
                    received_count: state.received_count,
                    last_offset: state.last_packet_id as u32,
                    last_chunk_idx: state.last_packet_id as u32,
                }
            })
            .collect();
        let chunk_bytes = self
            .images
            .first()
            .map(|state| state.packet_kind.payload_len() as u32)
            .unwrap_or(SsdvPacketKind::WithFec.payload_len() as u32);
        let (width, height) = self
            .images
            .first()
            .map(|state| (state.width as u32, state.height as u32))
            .unwrap_or((0, 0));
        ImageSnapshot {
            decoder: ImageDecoder::Jpeg,
            width,
            height,
            pixel_format: PixelFormat::Gray8,
            chunk_bytes,
            images,
        }
    }

    fn reset(&mut self) {
        self.images.clear();
        self.next_image_idx = 0;
    }
}

fn build_jpeg(state: &SsdvImageState) -> Vec<u8> {
    if state.packets.is_empty() {
        return Vec::new();
    }
    let mut builder = JpegBuilder::new();
    for p in state.packets.iter().flatten() {
        builder.feed_packet(p);
    }
    builder.finish()
}

#[cfg(test)]
mod ssdv_image_tests {
    use super::*;
    use crate::rs8;
    use crate::ssdv::{encode_callsign, SSDV_PKT_SIZE};

    fn build_packet(image_id: u8, packet_id: u16, payload_seed: u8) -> [u8; SSDV_PKT_SIZE] {
        let mut pkt = [0u8; SSDV_PKT_SIZE];
        pkt[0] = 0x55;
        pkt[1] = 0x66;
        let cs = encode_callsign("HSCAT1");
        pkt[2..6].copy_from_slice(&cs.to_be_bytes());
        pkt[6] = image_id;
        pkt[7..9].copy_from_slice(&packet_id.to_be_bytes());
        pkt[9] = 20; // 320 px
        pkt[10] = 15; // 240 px
                      // flags byte: ((quality ^ 4) << 3) | (eoi << 2) | mcu_mode.
                      // For quality=4, mcu_mode=0, eoi=0 this is zero.
        pkt[11] = 0;
        pkt[12] = 0;
        pkt[13..15].copy_from_slice(&packet_id.to_be_bytes());
        // Payload: 205 bytes of pseudo-random JPEG-ish bytes that
        // never contain 0xFF (which would trip JPEG marker logic).
        for (i, b) in pkt[15..15 + 205].iter_mut().enumerate() {
            *b = payload_seed.wrapping_add(i as u8) & 0x7F;
        }
        // CRC32 over [type..payload_end-1]
        let crcdata_len = 0x0F + 205 - 1;
        let crc = crate::ssdv::tests_helpers::crc32(&pkt[1..1 + crcdata_len]);
        let i = 1 + crcdata_len;
        pkt[i..i + 4].copy_from_slice(&crc.to_be_bytes());
        // RS parity
        let mut block = [0u8; 255];
        block[..223].copy_from_slice(&pkt[1..1 + 223]);
        rs8::encode(&mut block).expect("rs encode");
        pkt[1 + 223..1 + 255].copy_from_slice(&block[223..255]);
        pkt
    }

    #[test]
    fn assembles_complete_jpeg_with_three_packets() {
        let mut r = SsdvImageReassembler::new(None);
        for pid in 0..3u16 {
            let pkt = build_packet(7, pid, 0x11_u8.wrapping_add(pid as u8 * 5));
            r.ingest(&pkt).expect("ingest");
        }
        let snap = r.snapshot();
        assert_eq!(snap.images.len(), 1);
        let img = &snap.images[0];
        assert!(!img.bytes.is_empty());
        assert_eq!(&img.bytes[..2], &[0xFF, 0xD8]);
        assert_eq!(&img.bytes[img.bytes.len() - 2..], &[0xFF, 0xD9]);
        assert_eq!(img.received_count, 3);
        assert_eq!(img.total_chunks, 3);
    }

    #[test]
    fn missing_middle_packet_still_produces_jpeg() {
        let mut r = SsdvImageReassembler::new(None);
        for pid in [0u16, 2u16] {
            let pkt = build_packet(9, pid, 0x33);
            r.ingest(&pkt).expect("ingest");
        }
        let snap = r.snapshot();
        let img = &snap.images[0];
        assert!(!img.bytes.is_empty());
        assert_eq!(&img.bytes[..2], &[0xFF, 0xD8]);
        assert_eq!(&img.bytes[img.bytes.len() - 2..], &[0xFF, 0xD9]);
        assert_eq!(img.received_count, 2);
        assert_eq!(img.total_chunks, 3);
    }

    #[test]
    fn separate_image_ids_become_separate_streams() {
        let mut r = SsdvImageReassembler::new(None);
        let _ = r.ingest(&build_packet(1, 0, 0xAA));
        let _ = r.ingest(&build_packet(2, 0, 0xBB));
        let snap = r.snapshot();
        assert_eq!(snap.images.len(), 2);
    }

    #[test]
    fn callsign_filter_rejects_mismatch() {
        let mut r = SsdvImageReassembler::new(Some("OTHER1".to_string()));
        let pkt = build_packet(5, 0, 0x55);
        assert!(r.ingest(&pkt).is_none());
        assert!(r.snapshot().images.is_empty());
    }
}
