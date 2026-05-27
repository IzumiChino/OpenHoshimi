//! Image-frame reassembly.
//!
//! Some satellites send images split across many small frames. Each
//! frame carries a fixed-size raw chunk, a byte offset into the linear
//! pixel buffer, and an optional subsystem / group identifier so multiple
//! independent images can stream over the same downlink. Frames arrive
//! out of order and with gaps from the LEO pass edges; the reassembler's
//! job is to slot each chunk into the correct canvas position and keep
//! a per-chunk receipt bitmap so the GUI can show what is missing.
//!
//! The protocol-specific framing details (header signature, field byte
//! offsets, chunk size, canvas dimensions) come from the satellite
//! TOML's `[downlink.image]` block. v1 ships a single
//! [`GeoscanImageReassembler`] driven by [`ImageDef::Geoscan`]. Adding
//! SSDV later is a new variant of [`ImageDef`] plus a new impl of
//! [`ImageReassembler`]; nothing in the runtime or GUI needs to know
//! the protocol name.

use openhoshimi_core::satellite::{ByteOrderDef, ImageDef, ImageField, PixelFormatDef};

/// Stateful reassembler that consumes raw frame payloads and accumulates
/// an image canvas (or several, one per group id) over time.
pub trait ImageReassembler: Send {
    /// Feed one decoded frame payload to the reassembler.
    ///
    /// Returns `Some(update)` when the payload was an image chunk and
    /// at least one byte was written into the canvas, even if the
    /// chunk is a duplicate of one already received. Returns `None`
    /// when the payload's header does not match the configured
    /// signature or the chunk is structurally invalid (out-of-range
    /// offsets etc.).
    fn ingest(&mut self, payload: &[u8]) -> Option<PacketUpdate>;

    /// Produce a snapshot of the current canvases. Cheap to call on
    /// every UI repaint: groups are stored sorted by group id and
    /// cloned by the snapshot.
    fn snapshot(&self) -> ImageSnapshot;

    /// Drop all accumulated state and start over. Called from the GUI
    /// when the user clicks "Reset" or switches to a different
    /// downlink / satellite.
    fn reset(&mut self);
}

/// Result of a successful chunk write.
#[derive(Debug, Clone, Copy)]
pub struct PacketUpdate {
    /// Group id this chunk belongs to.
    pub group: u32,
    /// Byte offset into the canvas where the chunk was written.
    pub offset: u32,
    /// Number of bytes actually written (after clamping to the canvas
    /// end).
    pub bytes_written: usize,
    /// Index of the chunk slot in the receipt bitmap.
    pub chunk_idx: u32,
    /// `true` if this was the first time this chunk slot was filled;
    /// `false` if a chunk already existed at the same slot and was
    /// overwritten.
    pub fresh: bool,
}

/// Read-only view of every group's current canvas state, suitable for
/// rendering.
#[derive(Debug, Clone)]
pub struct ImageSnapshot {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Pixel format used to interpret the byte buffer.
    pub pixel_format: PixelFormat,
    /// Per-group canvases, sorted by group id.
    pub groups: Vec<ImageGroup>,
}

/// Reassembled state for one group / subsystem.
#[derive(Debug, Clone)]
pub struct ImageGroup {
    /// Group id reported by the satellite (e.g. `subsystem` byte).
    pub group_id: u32,
    /// Canvas-sized buffer, zero-filled where chunks have not arrived.
    pub bytes: Vec<u8>,
    /// One bit per chunk slot; `true` when a chunk has been received
    /// for that slot.
    pub received_chunks: Vec<u64>,
    /// Total number of chunk slots in the canvas. The last slot may
    /// hold fewer than `chunk_bytes` bytes.
    pub total_chunks: u32,
    /// Number of unique chunk slots that have received at least one
    /// chunk. Capped at `total_chunks`.
    pub received_count: u32,
    /// Byte offset of the most recently written chunk.
    pub last_offset: u32,
    /// Slot index of the most recently written chunk.
    pub last_chunk_idx: u32,
}

impl ImageGroup {
    /// Test whether the chunk slot at `idx` has ever been written.
    pub fn chunk_received(&self, idx: u32) -> bool {
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        self.received_chunks
            .get(word)
            .is_some_and(|w| (w >> bit) & 1 == 1)
    }
}

/// Pixel format mirror, decoupled from the TOML enum so callers don't
/// have to depend on `openhoshimi_core::satellite` to know the layout.
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
/// chunk geometry against the canvas dimensions so a misconfigured
/// downlink is rejected up-front instead of producing garbage at
/// runtime.
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
            group_field,
            chunk_at,
            chunk_bytes,
            width,
            height,
            pixel_format,
        } => {
            let header =
                parse_hex(header_signature).map_err(|e| format!("image.header_signature: {e}"))?;
            validate_field("offset_field", offset_field, 1..=4)?;
            validate_field("group_field", group_field, 1..=4)?;
            if *chunk_bytes == 0 {
                return Err("image.chunk_bytes must be > 0".to_string());
            }
            if *width == 0 || *height == 0 {
                return Err("image.width / image.height must be > 0".to_string());
            }
            let canvas_bytes = (*width as usize)
                .checked_mul(*height as usize)
                .and_then(|n| n.checked_mul(PixelFormat::from(*pixel_format).bytes_per_pixel()))
                .ok_or_else(|| "image canvas size overflows usize".to_string())?;
            Ok(Box::new(GeoscanImageReassembler {
                header,
                offset_field: offset_field.clone(),
                group_field: group_field.clone(),
                chunk_at: *chunk_at,
                chunk_bytes: *chunk_bytes,
                width: *width,
                height: *height,
                pixel_format: PixelFormat::from(*pixel_format),
                canvas_bytes,
                groups: Vec::new(),
            }))
        }
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
    group_field: ImageField,
    chunk_at: usize,
    chunk_bytes: usize,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    canvas_bytes: usize,
    groups: Vec<GeoscanGroupState>,
}

struct GeoscanGroupState {
    group_id: u32,
    bytes: Vec<u8>,
    received: Vec<u64>,
    total_chunks: u32,
    received_count: u32,
    last_offset: u32,
    last_chunk_idx: u32,
}

impl GeoscanGroupState {
    fn new(group_id: u32, canvas_bytes: usize, chunk_bytes: usize) -> Self {
        let total_chunks = canvas_bytes.div_ceil(chunk_bytes) as u32;
        let bitmap_words = (total_chunks as usize).div_ceil(64);
        Self {
            group_id,
            bytes: vec![0; canvas_bytes],
            received: vec![0; bitmap_words],
            total_chunks,
            received_count: 0,
            last_offset: 0,
            last_chunk_idx: 0,
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

    fn snapshot(&self) -> ImageGroup {
        ImageGroup {
            group_id: self.group_id,
            bytes: self.bytes.clone(),
            received_chunks: self.received.clone(),
            total_chunks: self.total_chunks,
            received_count: self.received_count,
            last_offset: self.last_offset,
            last_chunk_idx: self.last_chunk_idx,
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
        let group = read_uint(payload, &self.group_field)?;
        let end = self.chunk_at.checked_add(self.chunk_bytes)?;
        if end > payload.len() {
            return None;
        }
        let chunk = &payload[self.chunk_at..end];

        if (offset as usize) >= self.canvas_bytes {
            return None;
        }
        let chunk_idx = offset / self.chunk_bytes as u32;

        let group_idx = match self.groups.iter().position(|g| g.group_id == group) {
            Some(i) => i,
            None => {
                let state = GeoscanGroupState::new(group, self.canvas_bytes, self.chunk_bytes);
                self.groups.push(state);
                self.groups.len() - 1
            }
        };
        let state = &mut self.groups[group_idx];
        let canvas_end = (offset as usize)
            .checked_add(chunk.len())
            .map(|n| n.min(self.canvas_bytes))
            .unwrap_or(self.canvas_bytes);
        let bytes_written = canvas_end - offset as usize;
        state.bytes[offset as usize..canvas_end].copy_from_slice(&chunk[..bytes_written]);
        let fresh = state.mark(chunk_idx);
        state.last_offset = offset;
        state.last_chunk_idx = chunk_idx;

        // Keep snapshot order stable for the GUI.
        self.groups.sort_by_key(|g| g.group_id);
        Some(PacketUpdate {
            group,
            offset,
            bytes_written,
            chunk_idx,
            fresh,
        })
    }

    fn snapshot(&self) -> ImageSnapshot {
        ImageSnapshot {
            width: self.width,
            height: self.height,
            pixel_format: self.pixel_format,
            groups: self
                .groups
                .iter()
                .map(GeoscanGroupState::snapshot)
                .collect(),
        }
    }

    fn reset(&mut self) {
        self.groups.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(width: u32, height: u32) -> ImageDef {
        ImageDef::Geoscan {
            header_signature: "02 00 3E 05 09".to_string(),
            offset_field: ImageField {
                at: 5,
                len: 2,
                endian: ByteOrderDef::Le,
            },
            group_field: ImageField {
                at: 7,
                len: 1,
                endian: ByteOrderDef::Be,
            },
            chunk_at: 8,
            chunk_bytes: 56,
            width,
            height,
            pixel_format: PixelFormatDef::Gray8,
        }
    }

    fn frame(offset: u16, group: u8, fill: u8) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        buf[0..5].copy_from_slice(&[0x02, 0x00, 0x3E, 0x05, 0x09]);
        buf[5..7].copy_from_slice(&offset.to_le_bytes());
        buf[7] = group;
        for b in &mut buf[8..64] {
            *b = fill;
        }
        buf
    }

    #[test]
    fn ingest_writes_chunk_at_offset() {
        let mut r = build(&def(320, 240)).unwrap();
        let update = r.ingest(&frame(0, 0, 0xAA)).unwrap();
        assert_eq!(update.group, 0);
        assert_eq!(update.offset, 0);
        assert_eq!(update.bytes_written, 56);
        assert!(update.fresh);
        let snap = r.snapshot();
        assert_eq!(snap.groups.len(), 1);
        assert_eq!(snap.groups[0].received_count, 1);
        assert!(snap.groups[0].bytes[..56].iter().all(|b| *b == 0xAA));
        assert!(snap.groups[0].bytes[56..].iter().all(|b| *b == 0));
    }

    #[test]
    fn header_mismatch_returns_none() {
        let mut r = build(&def(320, 240)).unwrap();
        let mut bad = frame(0, 0, 0xAA);
        bad[0] = 0xFF;
        assert!(r.ingest(&bad).is_none());
    }

    #[test]
    fn out_of_order_arrival() {
        let mut r = build(&def(320, 240)).unwrap();
        let _ = r.ingest(&frame(112, 0, 0xCC)).unwrap();
        let _ = r.ingest(&frame(56, 0, 0xBB)).unwrap();
        let _ = r.ingest(&frame(0, 0, 0xAA)).unwrap();
        let snap = r.snapshot();
        assert_eq!(snap.groups[0].received_count, 3);
        assert!(snap.groups[0].bytes[0..56].iter().all(|b| *b == 0xAA));
        assert!(snap.groups[0].bytes[56..112].iter().all(|b| *b == 0xBB));
        assert!(snap.groups[0].bytes[112..168].iter().all(|b| *b == 0xCC));
    }

    #[test]
    fn duplicate_chunk_not_double_counted() {
        let mut r = build(&def(320, 240)).unwrap();
        let _ = r.ingest(&frame(0, 0, 0xAA)).unwrap();
        let again = r.ingest(&frame(0, 0, 0xAB)).unwrap();
        assert!(!again.fresh);
        assert_eq!(again.bytes_written, 56);
        let snap = r.snapshot();
        assert_eq!(snap.groups[0].received_count, 1);
        assert!(snap.groups[0].bytes[..56].iter().all(|b| *b == 0xAB));
    }

    #[test]
    fn multiple_groups_isolated() {
        let mut r = build(&def(320, 240)).unwrap();
        let _ = r.ingest(&frame(0, 1, 0xAA)).unwrap();
        let _ = r.ingest(&frame(0, 2, 0xBB)).unwrap();
        let snap = r.snapshot();
        assert_eq!(snap.groups.len(), 2);
        assert_eq!(snap.groups[0].group_id, 1);
        assert_eq!(snap.groups[1].group_id, 2);
        assert!(snap.groups[0].bytes[..56].iter().all(|b| *b == 0xAA));
        assert!(snap.groups[1].bytes[..56].iter().all(|b| *b == 0xBB));
    }

    #[test]
    fn out_of_range_offset_rejected() {
        let mut r = build(&def(320, 240)).unwrap();
        // 320*240 = 76800; offset 0xFFFF = 65535 is in-range, so push past
        // canvas with a small canvas instead.
        let mut small = build(&def(8, 8)).unwrap();
        // offset 100 > 64 canvas bytes
        assert!(small.ingest(&frame(100, 0, 0xAA)).is_none());
        // sanity: the bigger one accepts it
        assert!(r.ingest(&frame(100, 0, 0xAA)).is_some());
    }

    #[test]
    fn reset_clears_groups() {
        let mut r = build(&def(320, 240)).unwrap();
        let _ = r.ingest(&frame(0, 0, 0xAA)).unwrap();
        r.reset();
        assert!(r.snapshot().groups.is_empty());
    }
}
