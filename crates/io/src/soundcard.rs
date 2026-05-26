//! Live soundcard input source.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Data, FromSample, Sample, SampleFormat, Stream, StreamConfig, I24};
use openhoshimi_core::{InputSource, IoError};

const DEFAULT_BUFFER_SAMPLES: usize = 48_000;

/// Live soundcard input source backed by `cpal`.
pub struct SoundcardSource {
    state: Arc<(Mutex<SoundcardState>, Condvar)>,
    _stream: Stream,
    sample_rate: u32,
    description: String,
    closed: bool,
}

struct SoundcardState {
    buffer: VecDeque<f32>,
    closed: bool,
    error: Option<IoError>,
}

impl SoundcardSource {
    /// Open the default input device using its default input configuration.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Format`] if no default input device exists, the
    /// device has no supported input configuration, or the input stream
    /// cannot be built.
    pub fn open_default() -> Result<Self, IoError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| IoError::Format("no default input device available".to_string()))?;
        let supported = device.default_input_config().map_err(map_cpal_error)?;
        let sample_rate = supported.sample_rate().0;
        let sample_format = supported.sample_format();
        let channels = usize::from(supported.channels());
        let description = device.name().map_err(map_cpal_error)?;
        Self::open_device(
            device,
            supported.into(),
            sample_format,
            channels,
            description,
            sample_rate,
        )
    }

    /// Open a specific input device with a chosen stream config.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Format`] if the input stream cannot be created.
    pub fn open_device(
        device: cpal::Device,
        config: StreamConfig,
        sample_format: SampleFormat,
        channels: usize,
        description: String,
        sample_rate: u32,
    ) -> Result<Self, IoError> {
        let state = Arc::new((
            Mutex::new(SoundcardState {
                buffer: VecDeque::with_capacity(DEFAULT_BUFFER_SAMPLES),
                closed: false,
                error: None,
            }),
            Condvar::new(),
        ));
        let stream = build_stream(
            &device,
            &config,
            sample_format,
            channels,
            Arc::clone(&state),
        )?;
        stream.play().map_err(map_cpal_error)?;

        Ok(Self {
            state,
            _stream: stream,
            sample_rate,
            description: format!("soundcard {description} ({sample_rate} Hz, {channels} ch)"),
            closed: false,
        })
    }
}

impl InputSource for SoundcardSource {
    fn read_samples(&mut self, buf: &mut [f32]) -> Result<usize, IoError> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.closed {
            return Err(IoError::EndOfStream);
        }

        let (lock, cvar) = &*self.state;
        let mut guard = lock
            .lock()
            .map_err(|_| IoError::Format("soundcard buffer lock poisoned".to_string()))?;

        while guard.buffer.is_empty() && !guard.closed && guard.error.is_none() {
            guard = cvar
                .wait(guard)
                .map_err(|_| IoError::Format("soundcard wait failed".to_string()))?;
        }

        if let Some(err) = guard.error.take() {
            self.closed = true;
            return Err(err);
        }
        if guard.buffer.is_empty() && guard.closed {
            self.closed = true;
            return Err(IoError::EndOfStream);
        }

        let mut written = 0usize;
        while written < buf.len() {
            let Some(sample) = guard.buffer.pop_front() else {
                break;
            };
            buf[written] = sample;
            written += 1;
        }

        if written == 0 {
            self.closed = true;
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

fn build_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    channels: usize,
    state: Arc<(Mutex<SoundcardState>, Condvar)>,
) -> Result<Stream, IoError> {
    let err_state = Arc::clone(&state);
    let err_callback = move |err: cpal::StreamError| {
        let (lock, cvar) = &*err_state;
        if let Ok(mut guard) = lock.lock() {
            if guard.error.is_none() {
                guard.error = Some(IoError::Format(format!("soundcard stream error: {err}")));
            }
            guard.closed = true;
            cvar.notify_all();
        }
    };

    let data_state = Arc::clone(&state);
    let stream = device
        .build_input_stream_raw(
            config,
            sample_format,
            move |data: &Data, _| match sample_format {
                SampleFormat::I8 => {
                    if let Some(samples) = data.as_slice::<i8>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::I16 => {
                    if let Some(samples) = data.as_slice::<i16>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::I24 => {
                    if let Some(samples) = data.as_slice::<I24>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::I32 => {
                    if let Some(samples) = data.as_slice::<i32>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::I64 => {
                    if let Some(samples) = data.as_slice::<i64>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::U8 => {
                    if let Some(samples) = data.as_slice::<u8>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::U16 => {
                    if let Some(samples) = data.as_slice::<u16>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::U32 => {
                    if let Some(samples) = data.as_slice::<u32>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::U64 => {
                    if let Some(samples) = data.as_slice::<u64>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::F32 => {
                    if let Some(samples) = data.as_slice::<f32>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                SampleFormat::F64 => {
                    if let Some(samples) = data.as_slice::<f64>() {
                        push_samples(samples, channels, Arc::clone(&data_state));
                    }
                }
                _ => {}
            },
            err_callback,
            Some(Duration::from_millis(100)),
        )
        .map_err(map_cpal_error)?;

    Ok(stream)
}

fn push_samples<T>(data: &[T], channels: usize, state: Arc<(Mutex<SoundcardState>, Condvar)>)
where
    T: Copy,
    f32: FromSample<T>,
{
    let (lock, cvar) = &*state;
    if let Ok(mut guard) = lock.lock() {
        let channels = channels.max(1);
        for (index, sample) in data.iter().copied().enumerate() {
            if index % channels != 0 {
                continue;
            }
            let sample = f32::from_sample(sample);
            if guard.buffer.len() == DEFAULT_BUFFER_SAMPLES {
                guard.buffer.pop_front();
            }
            guard.buffer.push_back(sample);
        }
        cvar.notify_all();
    }
}

fn map_cpal_error(err: impl std::fmt::Display) -> IoError {
    IoError::Format(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soundcard_state_buffer_is_fifo() {
        let state = Arc::new((
            Mutex::new(SoundcardState {
                buffer: VecDeque::new(),
                closed: false,
                error: None,
            }),
            Condvar::new(),
        ));
        push_samples(&[0.25f32, -0.25, 0.5, -0.5], 1, Arc::clone(&state));

        let (lock, _) = &*state;
        let guard = lock.lock().unwrap_or_else(|_| panic!("lock"));
        assert_eq!(guard.buffer.len(), 4);
        assert_eq!(guard.buffer[0], 0.25);
        assert_eq!(guard.buffer[1], -0.25);
        assert_eq!(guard.buffer[2], 0.5);
        assert_eq!(guard.buffer[3], -0.5);
    }
}
