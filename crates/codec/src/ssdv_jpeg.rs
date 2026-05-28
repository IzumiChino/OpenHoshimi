//! SSDV bitstream → JPEG re-encoder.
//!
//! SSDV packet payloads are not raw JPEG entropy data. They use the
//! standard JPEG Huffman tables but at every reset-MCU boundary the DC
//! coefficient is stored as an absolute value (so each packet decodes
//! independently of its predecessors). To produce a JPEG that an
//! off-the-shelf decoder accepts, we Huffman-decode the SSDV bitstream
//! one packet at a time, convert each absolute DC back to the standard
//! differential form, and re-emit the result with JPEG byte stuffing.
//!
//! The algorithm follows fsphil's `ssdv.c` (`ssdv_process`,
//! `ssdv_fill_gap`, `ssdv_out_headers`, `ssdv_dec_feed`,
//! `ssdv_dec_get_jpeg`) line for line. Missing packets are filled with
//! zero-valued DC + EOB AC codes the same way the reference decoder
//! does — no JPEG restart markers are emitted, just a continuous scan
//! whose gaps are silent grey blocks.

use crate::ssdv::SsdvPacket;

const J_SOI: u16 = 0xFFD8;
const J_APP0: u16 = 0xFFE0;
const J_DQT: u16 = 0xFFDB;
const J_SOF0: u16 = 0xFFC0;
const J_DHT: u16 = 0xFFC4;
const J_SOS: u16 = 0xFFDA;
const J_EOI: u16 = 0xFFD9;

/// JFIF APP0 segment data (minus the 2-byte length field). From
/// `app0[14]` in fsphil/ssdv `ssdv.c`.
const APP0_DATA: [u8; 14] = [
    0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00,
];

/// SOS header data (3 components, Y/Cb/Cr). From `sos[10]`.
const SOS_DATA: [u8; 10] = [0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00];

/// Quantisation table scaling factors for each quality level 0..=7.
/// From `dqt_scales[8]`.
const DQT_SCALES: [u16; 8] = [5000, 357, 172, 116, 100, 58, 28, 0];

const STD_DQT0: [u8; 65] = [
    0x00, 0x10, 0x0C, 0x0C, 0x0E, 0x0C, 0x0A, 0x10, 0x0E, 0x0E, 0x0E, 0x12, 0x12, 0x10, 0x14, 0x18,
    0x28, 0x1A, 0x18, 0x16, 0x16, 0x18, 0x32, 0x24, 0x26, 0x1E, 0x28, 0x3A, 0x34, 0x3E, 0x3C, 0x3A,
    0x34, 0x38, 0x38, 0x40, 0x48, 0x5C, 0x4E, 0x40, 0x44, 0x58, 0x46, 0x38, 0x38, 0x50, 0x6E, 0x52,
    0x58, 0x60, 0x62, 0x68, 0x68, 0x68, 0x3E, 0x4E, 0x72, 0x7A, 0x70, 0x64, 0x78, 0x5C, 0x66, 0x68,
    0x64,
];
const STD_DQT1: [u8; 65] = [
    0x01, 0x12, 0x12, 0x12, 0x16, 0x16, 0x16, 0x30, 0x1A, 0x1A, 0x30, 0x64, 0x42, 0x38, 0x42, 0x64,
    0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64,
    0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64,
    0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64,
    0x64,
];

const STD_DHT00: [u8; 29] = [
    0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
];
const STD_DHT01: [u8; 29] = [
    0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
];
const STD_DHT10: [u8; 179] = [
    0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01,
    0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
    0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1,
    0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6,
    0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4,
    0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
    0xF8, 0xF9, 0xFA,
];
const STD_DHT11: [u8; 179] = [
    0x11, 0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02,
    0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
    0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52,
    0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A,
    0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4,
    0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2,
    0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
    0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
    0xF8, 0xF9, 0xFA,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcState {
    Huff,
    Int,
}

/// Builds a JPEG byte stream from a sequence of decoded SSDV packets.
///
/// Use [`JpegBuilder::feed_packet`] for each packet in `packet_id`
/// order (gaps are tolerated and filled), then call
/// [`JpegBuilder::finish`] to produce the complete JPEG.
pub struct JpegBuilder {
    out: Vec<u8>,
    out_bits: u32,
    out_len: u32,
    out_stuff: bool,

    state: ProcState,

    workbits: u32,
    worklen: u32,

    width: u16,
    height: u16,
    mcu_mode: u8,
    quality: u8,
    mcu_count: u32,
    ycparts: u8,

    mcu_id: u32,
    mcupart: u8,
    component: u8,
    acpart: u8,
    accrle: u8,
    acrle: u8,
    needbits: u8,

    dc: [i32; 3],

    reset_mcu: u32,
    next_reset_mcu: u32,

    headers_written: bool,
    expect_packet_id: u16,
}

impl JpegBuilder {
    /// Construct a new builder. The first call to [`feed_packet`]
    /// latches the image geometry and emits the JPEG headers.
    pub fn new() -> Self {
        Self {
            out: Vec::with_capacity(8192),
            out_bits: 0,
            out_len: 0,
            out_stuff: false,
            state: ProcState::Huff,
            workbits: 0,
            worklen: 0,
            width: 0,
            height: 0,
            mcu_mode: 0,
            quality: 0,
            mcu_count: 0,
            ycparts: 0,
            mcu_id: 0,
            mcupart: 0,
            component: 0,
            acpart: 0,
            accrle: 0,
            acrle: 0,
            needbits: 0,
            dc: [0; 3],
            reset_mcu: 0,
            next_reset_mcu: 0,
            headers_written: false,
            expect_packet_id: 0,
        }
    }

    /// Feed one decoded SSDV packet to the builder. Packets must be
    /// supplied in `packet_id` order; gaps trigger automatic
    /// fill-in. Bad bits inside a packet abort that packet's
    /// processing — the decoder resyncs at the next packet via the
    /// gap-fill path.
    pub fn feed_packet(&mut self, p: &SsdvPacket) {
        if !self.headers_written {
            self.write_headers(p);
            self.headers_written = true;
            self.out_stuff = true;
            // Match fsphil: the first call latches the expected
            // packet_id at 0 — if the actual packet_id is non-zero,
            // the gap-detection branch below fills the missing
            // prefix.
            self.expect_packet_id = 0;
        }

        let packet_id = p.packet_id;
        let packet_mcu_id = p.mcu_id;
        let packet_mcu_offset = p.mcu_offset;

        if packet_mcu_id != 0xFFFF {
            self.next_reset_mcu = packet_mcu_id as u32;
        }

        let mut i: usize = 0;

        if packet_id != self.expect_packet_id {
            if packet_id < self.expect_packet_id {
                // Out-of-order packet — drop.
                return;
            }
            // Gap detected. If the packet starts no fresh MCU we
            // can't pin the gap-fill target; skip entirely.
            if packet_mcu_id == 0xFFFF {
                return;
            }
            self.fill_gap(packet_mcu_id as u32);
            i = packet_mcu_offset as usize;
            self.state = ProcState::Huff;
            self.component = 0;
            self.mcupart = 0;
            self.acpart = 0;
            self.accrle = 0;
            self.expect_packet_id = packet_id;
        }

        let payload = &p.payload;
        while i < payload.len() {
            if i == packet_mcu_offset as usize {
                // First MCU in a packet is byte-aligned; drop any
                // residual bits and sanity-check mcu_id.
                self.workbits = 0;
                self.worklen = 0;
                if self.mcu_id != self.next_reset_mcu {
                    return;
                }
            }
            let b = payload[i];
            self.workbits = (self.workbits << 8) | b as u32;
            self.worklen += 8;
            loop {
                match self.process() {
                    StepResult::Ok => continue,
                    StepResult::FeedMe => break,
                    StepResult::Eoi => return,
                }
            }
            i += 1;
        }

        self.expect_packet_id = packet_id.wrapping_add(1);
    }

    /// Finish the JPEG: pad missing MCUs to `mcu_count`, sync the
    /// bit accumulator, disable byte stuffing, and emit EOI.
    pub fn finish(mut self) -> Vec<u8> {
        if !self.headers_written {
            return Vec::new();
        }
        if self.mcu_id < self.mcu_count {
            self.fill_gap(self.mcu_count);
        }
        self.outbits_sync();
        self.out_stuff = false;
        self.write_marker(J_EOI, &[]);
        self.out
    }

    fn write_headers(&mut self, p: &SsdvPacket) {
        self.width = p.width;
        self.height = p.height;
        self.mcu_mode = p.mcu_mode;
        self.quality = p.quality;
        let raw_mcu = (p.raw[9] as u32) * (p.raw[10] as u32);
        let (ycparts, mcu_count) = match self.mcu_mode & 3 {
            0 => (4u8, raw_mcu),
            1 => (2, raw_mcu * 2),
            2 => (2, raw_mcu * 2),
            3 => (1, raw_mcu * 4),
            _ => (4, raw_mcu),
        };
        self.ycparts = ycparts;
        self.mcu_count = mcu_count;

        let dqt0 = scale_dqt(&STD_DQT0, self.quality);
        let dqt1 = scale_dqt(&STD_DQT1, self.quality);

        self.write_marker(J_SOI, &[]);
        self.write_marker(J_APP0, &APP0_DATA);
        self.write_marker(J_DQT, &dqt0);
        self.write_marker(J_DQT, &dqt1);
        self.write_marker(J_SOF0, &self.sof0_segment());
        self.write_marker(J_DHT, &STD_DHT00);
        self.write_marker(J_DHT, &STD_DHT10);
        self.write_marker(J_DHT, &STD_DHT01);
        self.write_marker(J_DHT, &STD_DHT11);
        self.write_marker(J_SOS, &SOS_DATA);
    }

    fn sof0_segment(&self) -> [u8; 15] {
        let yh_yv = match self.mcu_mode {
            0 => 0x22,
            1 => 0x12,
            2 => 0x21,
            3 => 0x11,
            _ => 0x22,
        };
        [
            8,
            (self.height >> 8) as u8,
            (self.height & 0xFF) as u8,
            (self.width >> 8) as u8,
            (self.width & 0xFF) as u8,
            3,
            1,
            yh_yv,
            0x00,
            2,
            0x11,
            0x01,
            3,
            0x11,
            0x01,
        ]
    }

    fn process(&mut self) -> StepResult {
        match self.state {
            ProcState::Huff => self.process_huff(),
            ProcState::Int => self.process_int(),
        }
    }

    fn process_huff(&mut self) -> StepResult {
        if self.mcupart == 0 && self.acpart == 0 && self.next_reset_mcu > self.reset_mcu {
            self.reset_mcu = self.next_reset_mcu;
        }

        let dht = self.current_dht();
        let (symbol, width) = match jpeg_dht_lookup(dht, self.workbits, self.worklen as u8) {
            Some(v) => v,
            None => return StepResult::FeedMe,
        };

        if self.acpart == 0 {
            // DC
            if symbol == 0x00 {
                if self.reset_mcu == self.mcu_id
                    && (self.mcupart == 0 || self.mcupart >= self.ycparts)
                {
                    let comp = self.component as usize;
                    self.out_jpeg_int(0, 0 - self.dc[comp]);
                    self.dc[comp] = 0;
                } else {
                    self.out_jpeg_int(0, 0);
                }
                self.acpart += 1;
            } else {
                self.state = ProcState::Int;
                self.needbits = symbol;
            }
        } else {
            self.acrle = 0;
            if symbol == 0x00 {
                self.out_jpeg_int(0, 0);
                self.acpart = 64;
            } else if symbol == 0xF0 {
                self.out_jpeg_int(15, 0);
                self.acpart += 16;
            } else {
                self.state = ProcState::Int;
                self.acrle = symbol >> 4;
                self.acpart += self.acrle;
                self.needbits = symbol & 0x0F;
            }
        }

        self.worklen -= width as u32;
        self.workbits &= mask_lo(self.worklen);

        self.advance_mcu_if_complete()
    }

    fn process_int(&mut self) -> StepResult {
        if self.worklen < self.needbits as u32 {
            return StepResult::FeedMe;
        }
        let raw = (self.workbits >> (self.worklen - self.needbits as u32))
            & mask_lo(self.needbits as u32);
        let i = jpeg_int(raw as i32, self.needbits);

        if self.acpart == 0 {
            // DC
            let comp = self.component as usize;
            if self.reset_mcu == self.mcu_id && (self.mcupart == 0 || self.mcupart >= self.ycparts)
            {
                // SSDV stores absolute DC at reset; emit as relative
                self.out_jpeg_int(0, i - self.dc[comp]);
                self.dc[comp] = i;
            } else {
                self.dc[comp] += i;
                self.out_jpeg_int(0, i);
            }
        } else {
            // AC — DQTs are identical for src/dst so no rescale.
            if i != 0 {
                self.accrle += self.acrle;
                while self.accrle >= 16 {
                    self.out_jpeg_int(15, 0);
                    self.accrle -= 16;
                }
                self.out_jpeg_int(self.accrle, i);
                self.accrle = 0;
            } else if self.acpart >= 63 {
                self.out_jpeg_int(0, 0);
                self.accrle = 0;
            } else {
                self.accrle += self.acrle + 1;
            }
        }

        self.acpart += 1;
        self.state = ProcState::Huff;
        self.worklen -= self.needbits as u32;
        self.workbits &= mask_lo(self.worklen);

        self.advance_mcu_if_complete()
    }

    fn advance_mcu_if_complete(&mut self) -> StepResult {
        if self.acpart < 64 {
            return StepResult::Ok;
        }
        self.mcupart += 1;
        if self.mcupart == self.ycparts + 2 {
            self.mcupart = 0;
            self.mcu_id += 1;
            if self.mcu_id >= self.mcu_count {
                self.outbits_sync();
                return StepResult::Eoi;
            }
        }
        if self.mcupart < self.ycparts {
            self.component = 0;
        } else {
            self.component = self.mcupart - self.ycparts + 1;
        }
        self.acpart = 0;
        self.accrle = 0;
        StepResult::Ok
    }

    fn fill_gap(&mut self, next_mcu: u32) {
        if self.mcupart > 0 || self.acpart > 0 {
            if self.acpart > 0 {
                self.out_jpeg_int(0, 0);
                self.mcupart += 1;
            }
            while self.mcupart < self.ycparts + 2 {
                if self.mcupart < self.ycparts {
                    self.component = 0;
                } else {
                    self.component = self.mcupart - self.ycparts + 1;
                }
                self.acpart = 0;
                self.out_jpeg_int(0, 0);
                self.acpart = 1;
                self.out_jpeg_int(0, 0);
                self.mcupart += 1;
            }
            self.mcu_id += 1;
        }
        while self.mcu_id < next_mcu {
            self.mcupart = 0;
            while self.mcupart < self.ycparts + 2 {
                if self.mcupart < self.ycparts {
                    self.component = 0;
                } else {
                    self.component = self.mcupart - self.ycparts + 1;
                }
                self.acpart = 0;
                self.out_jpeg_int(0, 0);
                self.acpart = 1;
                self.out_jpeg_int(0, 0);
                self.mcupart += 1;
            }
            self.mcu_id += 1;
        }
        // Do NOT reset dc[]. fsphil's ssdv_fill_gap leaves the DC
        // predictors alone; the next real packet's first MCU is at
        // reset_mcu, where the absolute-DC branch emits
        // `i - dc[component]` as the JPEG delta and reseats dc[].
        // Clearing dc[] here would double-count the gap delta and
        // produce visible colour banding on either side of the
        // missing strip.
        self.mcupart = 0;
        self.acpart = 0;
        self.component = 0;
        self.accrle = 0;
    }

    fn current_dht(&self) -> &'static [u8] {
        // sdht[acpart ? 1 : 0][component ? 1 : 0] — src and dst use
        // the same standard tables, so a single helper covers both.
        match (self.acpart != 0, self.component != 0) {
            (false, false) => &STD_DHT00,
            (false, true) => &STD_DHT01,
            (true, false) => &STD_DHT10,
            (true, true) => &STD_DHT11,
        }
    }

    fn out_jpeg_int(&mut self, rle: u8, value: i32) {
        let (intbits, intlen) = jpeg_encode_int(value);
        let symbol = (rle << 4) | (intlen & 0x0F);
        if let Some((huff_bits, huff_len)) = jpeg_dht_encode(self.current_dht(), symbol) {
            self.outbits(huff_bits as u32, huff_len);
        }
        if intlen != 0 {
            self.outbits(intbits as u32, intlen);
        }
    }

    fn outbits(&mut self, bits: u32, length: u8) {
        if length > 0 {
            self.out_bits = (self.out_bits << length) | (bits & mask_lo(length as u32));
            self.out_len += length as u32;
        }
        while self.out_len >= 8 {
            let b = (self.out_bits >> (self.out_len - 8)) as u8;
            self.out.push(b);
            self.out_len -= 8;
            self.out_bits &= mask_lo(self.out_len);
            if self.out_stuff && b == 0xFF {
                // Re-emit a zero stuff byte by inflating outlen.
                self.out_len += 8;
            }
        }
    }

    fn outbits_sync(&mut self) {
        let r = self.out_len % 8;
        if r != 0 {
            self.outbits(0xFF, (8 - r) as u8);
        }
    }

    fn write_marker(&mut self, marker: u16, data: &[u8]) {
        let prev_stuff = self.out_stuff;
        self.out_stuff = false;
        self.outbits((marker >> 8) as u32, 8);
        self.outbits((marker & 0xFF) as u32, 8);
        if !data.is_empty() {
            let len = (data.len() as u16) + 2;
            self.outbits((len >> 8) as u32, 8);
            self.outbits((len & 0xFF) as u32, 8);
            for &b in data {
                self.outbits(b as u32, 8);
            }
        }
        self.out_stuff = prev_stuff;
    }
}

impl Default for JpegBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality-scale a 65-byte DQT (table-id + 64 zig-zag coefficients)
/// using the SSDV scale table. Mirrors fsphil's `load_standard_dqt`.
pub fn scale_dqt(table: &[u8; 65], quality: u8) -> [u8; 65] {
    let q = quality.min(7) as usize;
    let scale = DQT_SCALES[q] as u32;
    let mut out = [0u8; 65];
    out[0] = table[0];
    for i in 0..64 {
        let mut t = (table[i + 1] as u32 * scale + 50) / 100;
        if t == 0 {
            t = 1;
        }
        if t > 255 {
            t = 255;
        }
        out[i + 1] = t as u8;
    }
    out
}

/// Walk a JPEG-format DHT (1-byte class/id + 16 length counts +
/// symbols) to map the next codeword in `bits[..len]` to a symbol.
/// Returns `Some((symbol, codeword_width))` on a match, `None` if the
/// bit accumulator does not yet hold enough bits to disambiguate.
fn jpeg_dht_lookup(dht: &[u8], bits: u32, len: u8) -> Option<(u8, u8)> {
    let mut code: u32 = 0;
    let mut ss = 17usize;
    for cw in 1..=16u8 {
        if cw > len {
            return None;
        }
        let n = dht[cw as usize];
        for _ in 0..n {
            if (bits >> (len - cw)) & mask_lo(cw as u32) == code {
                return Some((dht[ss], cw));
            }
            ss += 1;
            code += 1;
        }
        code <<= 1;
    }
    // Out of range: matches fsphil's SSDV_ERROR. The caller treats
    // this as a packet abort; pretending we need more bits would
    // hang the loop.
    Some((0, 0))
}

/// Reverse lookup: encode `symbol` against `dht`, returning the
/// codeword bits and width. `None` if the table does not contain the
/// symbol (should not happen for correctly-paired src/dst tables).
fn jpeg_dht_encode(dht: &[u8], symbol: u8) -> Option<(u16, u8)> {
    let mut code: u16 = 0;
    let mut ss = 17usize;
    for cw in 1..=16u8 {
        let n = dht[cw as usize];
        for _ in 0..n {
            if dht[ss] == symbol {
                return Some((code, cw));
            }
            ss += 1;
            code += 1;
        }
        code <<= 1;
    }
    None
}

/// Convert raw bits + width into a signed JPEG coefficient.
fn jpeg_int(bits: i32, width: u8) -> i32 {
    if width == 0 {
        return 0;
    }
    let b = (1i32 << width) - 1;
    if bits <= b >> 1 {
        -(bits ^ b)
    } else {
        bits
    }
}

/// Encode a signed JPEG coefficient into (bits, width).
fn jpeg_encode_int(value: i32) -> (i32, u8) {
    let mut bits = value;
    let mut v = value.unsigned_abs();
    let mut width: u8 = 0;
    while v != 0 {
        width += 1;
        v >>= 1;
    }
    if bits < 0 {
        bits = -bits ^ ((1 << width) - 1);
    }
    (bits, width)
}

fn mask_lo(width: u32) -> u32 {
    if width >= 32 {
        u32::MAX
    } else if width == 0 {
        0
    } else {
        (1u32 << width) - 1
    }
}

#[derive(Debug, Clone, Copy)]
enum StepResult {
    Ok,
    FeedMe,
    Eoi,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jpeg_encode_int_round_trip() {
        for v in [-127i32, -1, 0, 1, 7, 64, 1023] {
            let (bits, width) = jpeg_encode_int(v);
            let back = jpeg_int(bits & ((1i32 << width.max(1)) - 1), width);
            assert_eq!(back, v, "value {v}");
        }
    }

    #[test]
    fn dht_lookup_round_trip_dht00() {
        // Symbol 0x05 should round-trip through encode/decode on the
        // DC luma table.
        let (bits, width) = jpeg_dht_encode(&STD_DHT00, 0x05).expect("encode");
        let (sym, w) = jpeg_dht_lookup(&STD_DHT00, bits as u32, width).expect("decode");
        assert_eq!(sym, 0x05);
        assert_eq!(w, width);
    }

    #[test]
    fn empty_builder_produces_empty_jpeg() {
        let jb = JpegBuilder::new();
        assert!(jb.finish().is_empty());
    }
}
