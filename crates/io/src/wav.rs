//! WAV file input source.
//!
//! Supports 8/16/24/32-bit integer PCM and 32-bit float PCM. Stereo files
//! use the left channel; multi-channel files take channel 0. All formats
//! are normalised to `f32` samples in `[-1.0, 1.0]`.

use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader};
use openhoshimi_core::{InputSource, IoError};

/// Boxed iterator yielding already-normalised, single-channel `f32`
/// samples. Hidden behind a trait object so the struct does not have to
/// be generic over hound's internal sample type.
type SampleIter = Box<dyn Iterator<Item = Result<f32, hound::Error>> + Send>;

/// Streaming WAV file reader implementing
/// [`openhoshimi_core::InputSource`].
///
/// Construct with [`WavSource::open`] and feed the resulting samples to a
/// [`openhoshimi_core::Demodulator`].
pub struct WavSource {
    iter: SampleIter,
    sample_rate: u32,
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
    /// format is not one of the supported variants.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let reader = WavReader::open(&path).map_err(map_hound_error)?;
        let spec = reader.spec();
        let channels = spec.channels.max(1) as usize;
        let sample_rate = spec.sample_rate;

        let description = format!(
            "WAV file {} ({} Hz, {}-bit {}, {} channel{})",
            path.display(),
            spec.sample_rate,
            spec.bits_per_sample,
            match spec.sample_format {
                SampleFormat::Int => "int",
                SampleFormat::Float => "float",
            },
            spec.channels,
            if spec.channels == 1 { "" } else { "s" },
        );

        let iter: SampleIter = match spec.sample_format {
            SampleFormat::Int => {
                let scale = int_scale(spec.bits_per_sample)?;
                Box::new(
                    reader
                        .into_samples::<i32>()
                        .enumerate()
                        .filter_map(move |(i, s)| {
                            if i % channels != 0 {
                                return None;
                            }
                            Some(s.map(|v| ((v as f32) / scale).clamp(-1.0, 1.0)))
                        }),
                )
            }
            SampleFormat::Float => {
                if spec.bits_per_sample != 32 {
                    return Err(IoError::Format(format!(
                        "unsupported float WAV: {}-bit (only 32-bit float is supported)",
                        spec.bits_per_sample
                    )));
                }
                Box::new(
                    reader
                        .into_samples::<f32>()
                        .enumerate()
                        .filter_map(move |(i, s)| {
                            if i % channels != 0 {
                                return None;
                            }
                            Some(s)
                        }),
                )
            }
        };

        Ok(Self {
            iter,
            sample_rate,
            description,
            eof: false,
        })
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
            match self.iter.next() {
                Some(Ok(v)) => {
                    *slot = v;
                    written += 1;
                }
                Some(Err(e)) => return Err(map_hound_error(e)),
                None => {
                    self.eof = true;
                    break;
                }
            }
        }
        if written == 0 {
            return Err(IoError::EndOfStream);
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

fn int_scale(bits_per_sample: u16) -> Result<f32, IoError> {
    match bits_per_sample {
        8 => Ok(128.0),
        16 => Ok(32_768.0),
        24 => Ok(8_388_608.0),
        32 => Ok(2_147_483_648.0),
        other => Err(IoError::Format(format!(
            "unsupported integer WAV bit depth: {other}"
        ))),
    }
}

fn map_hound_error(e: hound::Error) -> IoError {
    match e {
        hound::Error::IoError(io) => IoError::Io(io),
        other => IoError::Format(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openhoshimi-wav-test-{}", std::process::id()));
        if let Err(err) = std::fs::create_dir_all(&dir) {
            panic!("create temp dir: {err}");
        }
        dir.join(name)
    }

    fn write_wav_i16(path: &Path, channels: u16, samples: &[i16]) {
        let spec = WavSpec {
            channels,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = match WavWriter::create(path, spec) {
            Ok(writer) => writer,
            Err(err) => panic!("create writer: {err}"),
        };
        for s in samples {
            if let Err(err) = w.write_sample(*s) {
                panic!("write sample: {err}");
            }
        }
        if let Err(err) = w.finalize() {
            panic!("finalize writer: {err}");
        }
    }

    fn write_wav_f32(path: &Path, channels: u16, samples: &[f32]) {
        let spec = WavSpec {
            channels,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut w = match WavWriter::create(path, spec) {
            Ok(writer) => writer,
            Err(err) => panic!("create writer: {err}"),
        };
        for s in samples {
            if let Err(err) = w.write_sample(*s) {
                panic!("write sample: {err}");
            }
        }
        if let Err(err) = w.finalize() {
            panic!("finalize writer: {err}");
        }
    }

    #[test]
    fn reads_mono_i16_normalised() {
        let path = temp_path("mono16.wav");
        write_wav_i16(&path, 1, &[0, 16_384, -16_384, i16::MAX, i16::MIN]);

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

        // Next read should report end of stream.
        let err = match src.read_samples(&mut buf) {
            Ok(_) => panic!("eof"),
            Err(err) => err,
        };
        assert!(matches!(err, IoError::EndOfStream));
    }

    #[test]
    fn stereo_takes_left_channel_only() {
        let path = temp_path("stereo16.wav");
        // Interleaved L, R, L, R, ... - left channel is the ramp 0,1,2,3.
        let interleaved: Vec<i16> = (0..8)
            .flat_map(|i| [i as i16 * 1000, -(i as i16) * 1000])
            .collect();
        write_wav_i16(&path, 2, &interleaved);

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
    fn float32_passthrough() {
        let path = temp_path("mono_f32.wav");
        let samples = [-1.0f32, -0.25, 0.0, 0.25, 1.0];
        write_wav_f32(&path, 1, &samples);

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
        write_wav_i16(&path, 1, &[1234, -1234]);

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
}
