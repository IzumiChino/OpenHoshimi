//! OGG/Vorbis audio file input source.
//!
//! Reads mono audio from OGG Vorbis files (the format used by SatNOGS for
//! observation recordings). Multi-channel files are downmixed to mono by
//! taking the first channel.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use lewton::inside_ogg::OggStreamReader;
use openhoshimi_core::{InputSource, IoError};

/// Input source that reads audio samples from an OGG Vorbis file.
pub struct OggSource {
    reader: OggStreamReader<BufReader<File>>,
    sample_rate: u32,
    channels: u32,
    buffer: Vec<f32>,
    buffer_pos: usize,
    description: String,
}

impl OggSource {
    /// Open an OGG Vorbis file at the given path.
    pub fn open(path: &Path) -> Result<Self, IoError> {
        let file = File::open(path).map_err(IoError::Io)?;
        let reader = OggStreamReader::new(BufReader::new(file))
            .map_err(|e| IoError::Format(format!("failed to open OGG Vorbis stream: {e}")))?;

        let sample_rate = reader.ident_hdr.audio_sample_rate;
        let channels = u32::from(reader.ident_hdr.audio_channels);

        if sample_rate == 0 {
            return Err(IoError::Format(
                "OGG Vorbis file has zero sample rate".to_string(),
            ));
        }

        let description = format!(
            "OGG: {} ({} Hz, {} ch)",
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string()),
            sample_rate,
            channels,
        );

        Ok(Self {
            reader,
            sample_rate,
            channels,
            buffer: Vec::new(),
            buffer_pos: 0,
            description,
        })
    }

    /// Return the sample rate of the OGG file.
    pub fn sample_rate_value(&self) -> u32 {
        self.sample_rate
    }
}

impl InputSource for OggSource {
    fn read_samples(&mut self, buf: &mut [f32]) -> Result<usize, IoError> {
        let mut written = 0;

        while written < buf.len() {
            // Drain internal buffer first
            if self.buffer_pos < self.buffer.len() {
                let available = self.buffer.len() - self.buffer_pos;
                let to_copy = available.min(buf.len() - written);
                buf[written..written + to_copy]
                    .copy_from_slice(&self.buffer[self.buffer_pos..self.buffer_pos + to_copy]);
                self.buffer_pos += to_copy;
                written += to_copy;
                continue;
            }

            // Decode next packet
            match self.reader.read_dec_packet_itl() {
                Ok(Some(samples)) => {
                    // lewton returns interleaved i16-range samples as i16 vec
                    // Convert to f32 and take first channel only
                    self.buffer.clear();
                    self.buffer_pos = 0;
                    let ch = self.channels as usize;
                    for chunk in samples.chunks(ch) {
                        // Take first channel, normalize i16 to [-1.0, 1.0]
                        let sample = chunk[0] as f32 / 32768.0;
                        self.buffer.push(sample);
                    }
                }
                Ok(None) => {
                    // End of stream
                    if written == 0 {
                        return Err(IoError::EndOfStream);
                    }
                    break;
                }
                Err(e) => {
                    return Err(IoError::Format(format!("OGG Vorbis decode error: {e}")));
                }
            }
        }

        Ok(written)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn description(&self) -> &str {
        &self.description
    }
}
