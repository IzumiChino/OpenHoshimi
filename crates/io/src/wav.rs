//! WAV file input source.
//!
//! Supports classic RIFF WAVE (≤ 4 GiB) plus RF64/BW64 large-file
//! variants (EBU Tech 3306). Sample formats: 8/16/24/32-bit integer
//! PCM, 32-bit float PCM, and `WAVE_FORMAT_EXTENSIBLE` carrying either
//! of those via SubFormat GUID. Stereo files use the left channel for
//! [`WavSource`]; multi-channel files take channel 0. All samples are
//! normalised to `f32` in `[-1.0, 1.0]`.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use openhoshimi_core::{InputSource, IoError, IqSample, IqSource};

/// 16 KiB streaming read buffer for the underlying file.
const FILE_BUF: usize = 16 * 1024;

/// Marker in a classic RIFF size field meaning "see ds64 chunk".
const SIZE_SENTINEL: u32 = 0xFFFF_FFFF;

/// `WAVEFORMATEX::wFormatTag` values we care about.
const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// `KSDATAFORMAT_SUBTYPE_PCM`.
const SUBTYPE_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
/// `KSDATAFORMAT_SUBTYPE_IEEE_FLOAT`.
const SUBTYPE_IEEE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// Logical container kind sniffed from the leading 12 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    /// Classic 32-bit RIFF.
    Riff,
    /// RF64 — like RIFF but oversize, requires `ds64`.
    Rf64,
    /// BW64 — successor to RF64, layout identical for our purposes.
    Bw64,
}

/// Decoded sample format from the `fmt ` chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleFormat {
    /// Signed integer PCM. Width is `bits_per_sample`.
    Int,
    /// 32-bit IEEE 754 float PCM in `[-1.0, 1.0]`.
    Float,
}

/// Parsed WAV header summary.
#[derive(Debug, Clone)]
struct WavHeader {
    container: Container,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    block_align: u16,
    sample_format: SampleFormat,
    /// Number of bytes in the `data` chunk payload (post-RF64 fixup).
    data_bytes: u64,
}

impl WavHeader {
    fn frame_bytes(&self) -> u64 {
        // block_align is the size of one full channel frame; clamp to a sane
        // minimum derived from bits×channels in case the file lies.
        let derived = (self.bits_per_sample as u64).div_ceil(8) * self.channels as u64;
        let declared = self.block_align as u64;
        if declared == 0 {
            derived.max(1)
        } else {
            declared
        }
    }

    fn total_frames(&self) -> u64 {
        self.data_bytes.checked_div(self.frame_bytes()).unwrap_or(0)
    }
}

/// Streaming WAV file reader implementing
/// [`openhoshimi_core::InputSource`].
///
/// Construct with [`WavSource::open`] and feed the resulting samples to a
/// [`openhoshimi_core::Demodulator`]. Multi-channel inputs are folded to
/// channel 0.
pub struct WavSource {
    inner: SampleStream,
    sample_rate: u32,
    total_samples: Option<u64>,
    read_samples: u64,
    description: String,
    eof: bool,
}

/// Streaming stereo WAV IQ reader implementing
/// [`openhoshimi_core::IqSource`].
///
/// Channel 0 is interpreted as I, channel 1 as Q, and additional
/// channels are ignored. Sample formats supported by [`WavSource`] are
/// also accepted here.
pub struct WavIqSource {
    inner: SampleStream,
    sample_rate: u32,
    total_samples: Option<u64>,
    read_samples: u64,
    description: String,
    eof: bool,
}

impl WavSource {
    /// Open a WAV file for streaming reads.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Io`] if the file cannot be opened, or
    /// [`IoError::Format`] if the WAV header is invalid or its sample
    /// format is not supported.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let (header, file) = open_and_parse(&path)?;
        let sample_rate = header.sample_rate;
        let total_samples = Some(header.total_frames());
        let description = describe(&path, &header);
        let inner = SampleStream::new(file, header)?;
        Ok(Self {
            inner,
            sample_rate,
            total_samples,
            read_samples: 0,
            description,
            eof: false,
        })
    }

    /// Return the total number of logical samples in the file, if known.
    pub fn total_samples(&self) -> Option<u64> {
        self.total_samples
    }
}

impl WavIqSource {
    /// Open a stereo WAV file as complex IQ samples.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Io`] if the file cannot be opened, or
    /// [`IoError::Format`] if the header is invalid, the file has fewer
    /// than two channels, or the sample format is unsupported.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let (header, file) = open_and_parse(&path)?;
        if header.channels < 2 {
            return Err(IoError::Format(format!(
                "WAV IQ input requires at least 2 channels, got {}",
                header.channels
            )));
        }
        let sample_rate = header.sample_rate;
        let total_samples = Some(header.total_frames());
        let description = describe(&path, &header);
        let inner = SampleStream::new(file, header)?;
        Ok(Self {
            inner,
            sample_rate,
            total_samples,
            read_samples: 0,
            description,
            eof: false,
        })
    }

    /// Return the total number of logical samples in the file, if known.
    pub fn total_samples(&self) -> Option<u64> {
        self.total_samples
    }

    /// Return the number of logical samples already read from the file.
    pub fn read_samples(&self) -> u64 {
        self.read_samples
    }
}

impl InputSource for WavSource {
    fn read_samples(&mut self, buf: &mut [f32]) -> Result<usize, IoError> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.eof {
            return Err(IoError::EndOfStream);
        }
        let mut written = 0;
        for slot in buf.iter_mut() {
            match self.inner.next_frame()? {
                Some(frame) => {
                    *slot = frame[0];
                    written += 1;
                }
                None => {
                    self.eof = true;
                    break;
                }
            }
        }
        if written == 0 {
            return Err(IoError::EndOfStream);
        }
        self.read_samples += written as u64;
        Ok(written)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn total_samples(&self) -> Option<u64> {
        self.total_samples
    }
}

impl IqSource for WavIqSource {
    fn read_samples(&mut self, buf: &mut [IqSample]) -> Result<usize, IoError> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.eof {
            return Err(IoError::EndOfStream);
        }
        let mut written = 0;
        for slot in buf.iter_mut() {
            match self.inner.next_frame()? {
                Some(frame) => {
                    *slot = IqSample {
                        i: frame[0],
                        q: frame[1],
                    };
                    written += 1;
                }
                None => {
                    self.eof = true;
                    break;
                }
            }
        }
        if written == 0 {
            return Err(IoError::EndOfStream);
        }
        self.read_samples += written as u64;
        Ok(written)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn total_samples(&self) -> Option<u64> {
        self.total_samples
    }
}

/// Streaming sample decoder shared by mono and IQ readers.
struct SampleStream {
    file: BufReader<File>,
    header: WavHeader,
    bytes_left: u64,
    /// Reusable scratch buffer sized for one full channel frame.
    scratch: Vec<u8>,
    /// Reusable scratch buffer sized for the per-frame f32 output.
    out: Vec<f32>,
}

impl SampleStream {
    fn new(file: BufReader<File>, header: WavHeader) -> Result<Self, IoError> {
        let frame = usize::try_from(header.frame_bytes()).map_err(|_| {
            IoError::Format(format!(
                "WAV frame size {} exceeds usize",
                header.frame_bytes()
            ))
        })?;
        if frame == 0 {
            return Err(IoError::Format("WAV frame size is zero".to_string()));
        }
        let channels = header.channels as usize;
        let scratch = vec![0u8; frame];
        let out = vec![0f32; channels];
        Ok(Self {
            file,
            bytes_left: header.data_bytes,
            header,
            scratch,
            out,
        })
    }

    fn next_frame(&mut self) -> Result<Option<&[f32]>, IoError> {
        let frame = self.scratch.len() as u64;
        if self.bytes_left < frame {
            // Drain the stub if any so callers see consistent EOF.
            self.bytes_left = 0;
            return Ok(None);
        }
        match self.file.read_exact(&mut self.scratch) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(err) => return Err(IoError::Io(err)),
        }
        self.bytes_left -= frame;
        decode_frame(&self.scratch, &mut self.out, &self.header)?;
        Ok(Some(&self.out))
    }
}

fn decode_frame(bytes: &[u8], out: &mut [f32], header: &WavHeader) -> Result<(), IoError> {
    let channels = header.channels as usize;
    let bps = header.bits_per_sample;
    let bytes_per_sample = (bps as usize).div_ceil(8);
    if out.len() != channels {
        return Err(IoError::Format(format!(
            "output buffer length {} mismatches channel count {}",
            out.len(),
            channels
        )));
    }
    if bytes.len() < channels * bytes_per_sample {
        return Err(IoError::Format(format!(
            "frame buffer {} bytes < expected {}",
            bytes.len(),
            channels * bytes_per_sample
        )));
    }
    match (header.sample_format, bps) {
        (SampleFormat::Int, 8) => {
            // 8-bit PCM is unsigned, biased by 0x80 (per the spec).
            for (ch, slot) in out.iter_mut().enumerate().take(channels) {
                let v = bytes[ch] as i16 - 128;
                *slot = (v as f32 / 128.0).clamp(-1.0, 1.0);
            }
        }
        (SampleFormat::Int, 16) => {
            for (ch, slot) in out.iter_mut().enumerate().take(channels) {
                let lo = bytes[ch * 2] as u16;
                let hi = bytes[ch * 2 + 1] as u16;
                let v = (hi << 8 | lo) as i16;
                *slot = (v as f32 / 32_768.0).clamp(-1.0, 1.0);
            }
        }
        (SampleFormat::Int, 24) => {
            for (ch, slot) in out.iter_mut().enumerate().take(channels) {
                let b0 = bytes[ch * 3] as u32;
                let b1 = bytes[ch * 3 + 1] as u32;
                let b2 = bytes[ch * 3 + 2] as u32;
                let mut v = b0 | (b1 << 8) | (b2 << 16);
                if v & 0x0080_0000 != 0 {
                    v |= 0xFF00_0000;
                }
                let signed = v as i32;
                *slot = (signed as f32 / 8_388_608.0).clamp(-1.0, 1.0);
            }
        }
        (SampleFormat::Int, 32) => {
            for (ch, slot) in out.iter_mut().enumerate().take(channels) {
                let off = ch * 4;
                let arr = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
                let v = i32::from_le_bytes(arr);
                *slot = (v as f32 / 2_147_483_648.0).clamp(-1.0, 1.0);
            }
        }
        (SampleFormat::Float, 32) => {
            for (ch, slot) in out.iter_mut().enumerate().take(channels) {
                let off = ch * 4;
                let arr = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
                *slot = f32::from_le_bytes(arr);
            }
        }
        (fmt, bits) => {
            return Err(IoError::Format(format!(
                "unsupported sample format: {fmt:?} {bits}-bit"
            )));
        }
    }
    Ok(())
}

fn open_and_parse(path: &Path) -> Result<(WavHeader, BufReader<File>), IoError> {
    let file = File::open(path).map_err(IoError::Io)?;
    let mut reader = BufReader::with_capacity(FILE_BUF, file);
    let header = parse_header(&mut reader)?;
    Ok((header, reader))
}

fn parse_header(reader: &mut BufReader<File>) -> Result<WavHeader, IoError> {
    let mut hdr = [0u8; 12];
    reader.read_exact(&mut hdr).map_err(IoError::Io)?;

    let container = match &hdr[0..4] {
        b"RIFF" => Container::Riff,
        b"RF64" => Container::Rf64,
        b"BW64" => Container::Bw64,
        other => {
            return Err(IoError::Format(format!(
                "not a WAV file: leading tag {:?}",
                String::from_utf8_lossy(other)
            )));
        }
    };
    if &hdr[8..12] != b"WAVE" {
        return Err(IoError::Format(format!(
            "not a WAVE file: form-type {:?}",
            String::from_utf8_lossy(&hdr[8..12])
        )));
    }

    let mut ds64_data_size: Option<u64> = None;
    let mut ds64_sample_count: Option<u64> = None;
    let mut fmt_chunk: Option<Vec<u8>> = None;
    let mut data_size_u32: Option<u32> = None;
    let mut data_seen = false;

    loop {
        let mut chunk_hdr = [0u8; 8];
        match reader.read_exact(&mut chunk_hdr) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(IoError::Io(err)),
        }
        let id = [chunk_hdr[0], chunk_hdr[1], chunk_hdr[2], chunk_hdr[3]];
        let size_u32 = u32::from_le_bytes([chunk_hdr[4], chunk_hdr[5], chunk_hdr[6], chunk_hdr[7]]);
        let size = if size_u32 == SIZE_SENTINEL && container != Container::Riff {
            match &id {
                b"data" => ds64_data_size.ok_or_else(|| {
                    IoError::Format("RF64/BW64 data chunk requires ds64 sizes".to_string())
                })?,
                _ => u64::from(size_u32),
            }
        } else {
            u64::from(size_u32)
        };

        match &id {
            b"ds64" => {
                if container == Container::Riff {
                    return Err(IoError::Format(
                        "ds64 chunk found in classic RIFF WAV".to_string(),
                    ));
                }
                let mut ds64 = vec![0u8; size as usize];
                reader.read_exact(&mut ds64).map_err(IoError::Io)?;
                if ds64.len() < 24 {
                    return Err(IoError::Format(format!(
                        "ds64 chunk too short: {} bytes",
                        ds64.len()
                    )));
                }
                // bytes 0..8 = riffSize, 8..16 = dataSize, 16..24 = sampleCount, 24..28 = tableLength
                ds64_data_size = Some(u64::from_le_bytes(slice8(&ds64, 8)?));
                ds64_sample_count = Some(u64::from_le_bytes(slice8(&ds64, 16)?));
                pad_to_word(reader, size)?;
            }
            b"fmt " => {
                let mut buf = vec![0u8; size as usize];
                reader.read_exact(&mut buf).map_err(IoError::Io)?;
                fmt_chunk = Some(buf);
                pad_to_word(reader, size)?;
            }
            b"data" => {
                data_size_u32 = Some(size_u32);
                data_seen = true;
                // We don't seek past the data chunk — the stream sits at the
                // first sample byte and the sample reader takes over.
                break;
            }
            _ => {
                // Unknown chunk: skip its payload + word-align pad.
                if size > 0 {
                    reader.seek_relative(size as i64).map_err(IoError::Io)?;
                }
                pad_to_word(reader, size)?;
            }
        }
    }

    let fmt_bytes = fmt_chunk.ok_or_else(|| IoError::Format("missing fmt chunk".to_string()))?;
    let fmt = parse_fmt(&fmt_bytes)?;
    if !data_seen {
        return Err(IoError::Format("missing data chunk".to_string()));
    }
    let data_bytes = match (container, ds64_data_size, data_size_u32) {
        (Container::Riff, _, Some(s)) => u64::from(s),
        (_, Some(s), _) => s,
        (_, None, Some(SIZE_SENTINEL)) => {
            return Err(IoError::Format(
                "RF64/BW64 data size sentinel without ds64".to_string(),
            ));
        }
        (_, None, Some(s)) => u64::from(s),
        _ => {
            return Err(IoError::Format(
                "could not determine WAV data chunk size".to_string(),
            ));
        }
    };
    if let Some(declared) = ds64_sample_count {
        let frames_from_bytes = data_bytes / fmt.block_align.max(1) as u64;
        if declared != 0 && declared != frames_from_bytes {
            // The two should agree; if they disagree we trust dataSize since
            // sampleCount is sometimes zero in the wild.
            eprintln!(
                "openhoshimi-io: ds64 sampleCount={declared} disagrees with frame count {frames_from_bytes}; using dataSize"
            );
        }
    }

    Ok(WavHeader {
        container,
        sample_rate: fmt.sample_rate,
        channels: fmt.channels,
        bits_per_sample: fmt.bits_per_sample,
        block_align: fmt.block_align,
        sample_format: fmt.sample_format,
        data_bytes,
    })
}

struct FmtFields {
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    block_align: u16,
    sample_format: SampleFormat,
}

fn parse_fmt(bytes: &[u8]) -> Result<FmtFields, IoError> {
    if bytes.len() < 16 {
        return Err(IoError::Format(format!(
            "fmt chunk too short: {} bytes",
            bytes.len()
        )));
    }
    let format_tag = u16::from_le_bytes([bytes[0], bytes[1]]);
    let channels = u16::from_le_bytes([bytes[2], bytes[3]]);
    let sample_rate = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let block_align = u16::from_le_bytes([bytes[12], bytes[13]]);
    let bits_per_sample = u16::from_le_bytes([bytes[14], bytes[15]]);
    if channels == 0 {
        return Err(IoError::Format(
            "fmt chunk reports zero channels".to_string(),
        ));
    }
    if sample_rate == 0 {
        return Err(IoError::Format(
            "fmt chunk reports zero sample rate".to_string(),
        ));
    }

    let sample_format = match format_tag {
        WAVE_FORMAT_PCM => SampleFormat::Int,
        WAVE_FORMAT_IEEE_FLOAT => {
            if bits_per_sample != 32 {
                return Err(IoError::Format(format!(
                    "unsupported float WAV: {bits_per_sample}-bit (only 32-bit float is supported)"
                )));
            }
            SampleFormat::Float
        }
        WAVE_FORMAT_EXTENSIBLE => {
            if bytes.len() < 18 {
                return Err(IoError::Format(
                    "WAVE_FORMAT_EXTENSIBLE missing cbSize".to_string(),
                ));
            }
            let cb_size = u16::from_le_bytes([bytes[16], bytes[17]]) as usize;
            if cb_size < 22 || bytes.len() < 18 + 22 {
                return Err(IoError::Format(format!(
                    "WAVE_FORMAT_EXTENSIBLE extension too short: cb={cb_size}, total={}",
                    bytes.len()
                )));
            }
            // 18..20 validBitsPerSample, 20..24 channelMask, 24..40 SubFormat GUID
            let guid: [u8; 16] = slice16(bytes, 24)?;
            if guid == SUBTYPE_PCM {
                SampleFormat::Int
            } else if guid == SUBTYPE_IEEE_FLOAT {
                if bits_per_sample != 32 {
                    return Err(IoError::Format(format!(
                        "unsupported float WAV: {bits_per_sample}-bit (only 32-bit float is supported)"
                    )));
                }
                SampleFormat::Float
            } else {
                return Err(IoError::Format(format!(
                    "unsupported WAVE_FORMAT_EXTENSIBLE SubFormat GUID: {}",
                    hex_guid(&guid)
                )));
            }
        }
        other => {
            return Err(IoError::Format(format!(
                "unsupported WAVE format tag: 0x{other:04X}"
            )));
        }
    };

    if !matches!((sample_format, bits_per_sample), |(
        SampleFormat::Int,
        8 | 16 | 24 | 32,
    )| (
        SampleFormat::Float,
        32
    )) {
        return Err(IoError::Format(format!(
            "unsupported PCM bit depth: {bits_per_sample} (format {sample_format:?})"
        )));
    }

    Ok(FmtFields {
        sample_rate,
        channels,
        bits_per_sample,
        block_align,
        sample_format,
    })
}

fn pad_to_word(reader: &mut BufReader<File>, size: u64) -> Result<(), IoError> {
    if size & 1 == 1 {
        let mut pad = [0u8; 1];
        match reader.read_exact(&mut pad) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => Ok(()),
            Err(err) => Err(IoError::Io(err)),
        }
    } else {
        Ok(())
    }
}

fn slice8(bytes: &[u8], offset: usize) -> Result<[u8; 8], IoError> {
    bytes
        .get(offset..offset + 8)
        .and_then(|s| <[u8; 8]>::try_from(s).ok())
        .ok_or_else(|| {
            IoError::Format(format!(
                "chunk too short to read 8 bytes at offset {offset}"
            ))
        })
}

fn slice16(bytes: &[u8], offset: usize) -> Result<[u8; 16], IoError> {
    bytes
        .get(offset..offset + 16)
        .and_then(|s| <[u8; 16]>::try_from(s).ok())
        .ok_or_else(|| {
            IoError::Format(format!(
                "chunk too short to read 16 bytes at offset {offset}"
            ))
        })
}

fn hex_guid(guid: &[u8; 16]) -> String {
    let mut out = String::with_capacity(36);
    for (i, b) in guid.iter().enumerate() {
        if i == 4 || i == 6 || i == 8 || i == 10 {
            out.push('-');
        }
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn describe(path: &Path, header: &WavHeader) -> String {
    let kind = match header.container {
        Container::Riff => "WAV",
        Container::Rf64 => "RF64 WAV",
        Container::Bw64 => "BW64 WAV",
    };
    let fmt = match header.sample_format {
        SampleFormat::Int => "int",
        SampleFormat::Float => "float",
    };
    format!(
        "{kind} file {} ({} Hz, {}-bit {fmt}, {} channel{})",
        path.display(),
        header.sample_rate,
        header.bits_per_sample,
        header.channels,
        if header.channels == 1 { "" } else { "s" },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openhoshimi-wav-test-{}", std::process::id()));
        if let Err(err) = std::fs::create_dir_all(&dir) {
            panic!("create temp dir: {err}");
        }
        dir.join(name)
    }

    fn build_riff_pcm16(channels: u16, sample_rate: u32, samples_le: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(44 + samples_le.len());
        let block_align = channels * 2;
        let byte_rate = sample_rate * block_align as u32;
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36u32 + samples_le.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(samples_le.len() as u32).to_le_bytes());
        out.extend_from_slice(samples_le);
        out
    }

    fn build_riff_float32(channels: u16, sample_rate: u32, samples_le: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(46 + samples_le.len());
        let block_align = channels * 4;
        let byte_rate = sample_rate * block_align as u32;
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(38u32 + samples_le.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&18u32.to_le_bytes());
        out.extend_from_slice(&WAVE_FORMAT_IEEE_FLOAT.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // cbSize
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(samples_le.len() as u32).to_le_bytes());
        out.extend_from_slice(samples_le);
        out
    }

    fn build_rf64_pcm16(channels: u16, sample_rate: u32, samples_le: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let block_align = channels * 2;
        let byte_rate = sample_rate * block_align as u32;
        let sample_count = samples_le.len() as u64 / block_align as u64;
        out.extend_from_slice(b"RF64");
        out.extend_from_slice(&SIZE_SENTINEL.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        // ds64 chunk
        out.extend_from_slice(b"ds64");
        out.extend_from_slice(&28u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // riffSize (unused by reader)
        out.extend_from_slice(&(samples_le.len() as u64).to_le_bytes()); // dataSize
        out.extend_from_slice(&sample_count.to_le_bytes()); // sampleCount
        out.extend_from_slice(&0u32.to_le_bytes()); // tableLength
                                                    // fmt
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        // data with sentinel size
        out.extend_from_slice(b"data");
        out.extend_from_slice(&SIZE_SENTINEL.to_le_bytes());
        out.extend_from_slice(samples_le);
        out
    }

    fn build_extensible_pcm16(channels: u16, sample_rate: u32, samples_le: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let block_align = channels * 2;
        let byte_rate = sample_rate * block_align as u32;
        let fmt_size: u32 = 40;
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(20u32 + fmt_size + samples_le.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&fmt_size.to_le_bytes());
        out.extend_from_slice(&WAVE_FORMAT_EXTENSIBLE.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        out.extend_from_slice(&16u16.to_le_bytes()); // validBitsPerSample
        out.extend_from_slice(&0u32.to_le_bytes()); // channelMask
        out.extend_from_slice(&SUBTYPE_PCM);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(samples_le.len() as u32).to_le_bytes());
        out.extend_from_slice(samples_le);
        out
    }

    fn write_bytes(path: &Path, bytes: &[u8]) {
        let mut f = match File::create(path) {
            Ok(f) => f,
            Err(err) => panic!("create file: {err}"),
        };
        if let Err(err) = f.write_all(bytes) {
            panic!("write file: {err}");
        }
    }

    fn pcm16_le(samples: &[i16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    fn float32_le(samples: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 4);
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn reads_mono_i16_normalised() {
        let path = temp_path("mono16.wav");
        let payload = pcm16_le(&[0, 16_384, -16_384, i16::MAX, i16::MIN]);
        write_bytes(&path, &build_riff_pcm16(1, 48_000, &payload));

        let mut src = match WavSource::open(&path) {
            Ok(src) => src,
            Err(err) => panic!("open: {err}"),
        };
        assert_eq!(src.sample_rate(), 48_000);
        let mut buf = [0f32; 8];
        let n = match src.read_samples(&mut buf) {
            Ok(n) => n,
            Err(err) => panic!("read: {err}"),
        };
        assert_eq!(n, 5);
        assert!((buf[0] - 0.0).abs() < 1e-6);
        assert!((buf[1] - 0.5).abs() < 1e-4);
        assert!((buf[2] + 0.5).abs() < 1e-4);
        assert!(buf[3] > 0.999 && buf[3] < 1.0);
        assert!((buf[4] + 1.0).abs() < 1e-6);
        let err = match src.read_samples(&mut buf) {
            Ok(_) => panic!("eof"),
            Err(err) => err,
        };
        assert!(matches!(err, IoError::EndOfStream));
    }

    #[test]
    fn stereo_takes_left_channel_only() {
        let path = temp_path("stereo16.wav");
        let interleaved: Vec<i16> = (0..8)
            .flat_map(|i| [i as i16 * 1000, -(i as i16) * 1000])
            .collect();
        let payload = pcm16_le(&interleaved);
        write_bytes(&path, &build_riff_pcm16(2, 48_000, &payload));

        let mut src = match WavSource::open(&path) {
            Ok(src) => src,
            Err(err) => panic!("open: {err}"),
        };
        let mut buf = [0f32; 16];
        let n = match src.read_samples(&mut buf) {
            Ok(n) => n,
            Err(err) => panic!("read: {err}"),
        };
        assert_eq!(n, 8, "should drop the right channel");
        for (i, &s) in buf[..8].iter().enumerate() {
            let expected = (i as f32) * 1000.0 / 32_768.0;
            assert!(
                (s - expected).abs() < 1e-4,
                "left[{i}] = {s}, expected {expected}"
            );
        }
    }

    #[test]
    fn stereo_i16_reads_iq_pairs() {
        let path = temp_path("iq16.wav");
        let payload = pcm16_le(&[16_384, -16_384, -8192, 8192]);
        write_bytes(&path, &build_riff_pcm16(2, 48_000, &payload));

        let mut src = match WavIqSource::open(&path) {
            Ok(src) => src,
            Err(err) => panic!("open: {err}"),
        };
        let mut buf = [IqSample::default(); 4];
        let n = match IqSource::read_samples(&mut src, &mut buf) {
            Ok(n) => n,
            Err(err) => panic!("read: {err}"),
        };
        assert_eq!(n, 2);
        assert!((buf[0].i - 0.5).abs() < 1e-4);
        assert!((buf[0].q + 0.5).abs() < 1e-4);
        assert!((buf[1].i + 0.25).abs() < 1e-4);
        assert!((buf[1].q - 0.25).abs() < 1e-4);
    }

    #[test]
    fn mono_iq_is_rejected() {
        let path = temp_path("mono_iq.wav");
        let payload = pcm16_le(&[0, 1]);
        write_bytes(&path, &build_riff_pcm16(1, 48_000, &payload));

        let err = match WavIqSource::open(&path) {
            Ok(_) => panic!("mono IQ should fail"),
            Err(err) => err,
        };
        assert!(matches!(err, IoError::Format(_)));
    }

    #[test]
    fn float32_passthrough() {
        let path = temp_path("mono_f32.wav");
        let samples = [-1.0f32, -0.25, 0.0, 0.25, 1.0];
        let payload = float32_le(&samples);
        write_bytes(&path, &build_riff_float32(1, 48_000, &payload));

        let mut src = match WavSource::open(&path) {
            Ok(src) => src,
            Err(err) => panic!("open: {err}"),
        };
        let mut buf = [0f32; 8];
        let n = match src.read_samples(&mut buf) {
            Ok(n) => n,
            Err(err) => panic!("read: {err}"),
        };
        assert_eq!(n, samples.len());
        for (i, &s) in samples.iter().enumerate() {
            assert!((buf[i] - s).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_buffer_returns_zero() {
        let path = temp_path("mono16_short.wav");
        let payload = pcm16_le(&[1234, -1234]);
        write_bytes(&path, &build_riff_pcm16(1, 48_000, &payload));

        let mut src = match WavSource::open(&path) {
            Ok(src) => src,
            Err(err) => panic!("open: {err}"),
        };
        let mut buf: [f32; 0] = [];
        let n = match src.read_samples(&mut buf) {
            Ok(n) => n,
            Err(err) => panic!("read empty: {err}"),
        };
        assert_eq!(n, 0);
    }

    #[test]
    fn reads_rf64_iq() {
        let path = temp_path("rf64_iq.wav");
        let payload = pcm16_le(&[16_384, -16_384, -8192, 8192]);
        write_bytes(&path, &build_rf64_pcm16(2, 1_400_000, &payload));

        let mut src = match WavIqSource::open(&path) {
            Ok(src) => src,
            Err(err) => panic!("open RF64: {err}"),
        };
        assert_eq!(src.sample_rate(), 1_400_000);
        let mut buf = [IqSample::default(); 4];
        let n = match IqSource::read_samples(&mut src, &mut buf) {
            Ok(n) => n,
            Err(err) => panic!("read RF64: {err}"),
        };
        assert_eq!(n, 2);
        assert!((buf[0].i - 0.5).abs() < 1e-4);
        assert!((buf[0].q + 0.5).abs() < 1e-4);
    }

    #[test]
    fn reads_extensible_pcm() {
        let path = temp_path("extensible16.wav");
        let payload = pcm16_le(&[0, 16_384]);
        write_bytes(&path, &build_extensible_pcm16(1, 48_000, &payload));

        let mut src = match WavSource::open(&path) {
            Ok(src) => src,
            Err(err) => panic!("open EXTENSIBLE: {err}"),
        };
        let mut buf = [0f32; 4];
        let n = match src.read_samples(&mut buf) {
            Ok(n) => n,
            Err(err) => panic!("read: {err}"),
        };
        assert_eq!(n, 2);
        assert!((buf[0] - 0.0).abs() < 1e-6);
        assert!((buf[1] - 0.5).abs() < 1e-4);
    }
}
