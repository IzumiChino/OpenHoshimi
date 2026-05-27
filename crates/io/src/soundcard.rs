//! Live soundcard input source.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Data, FromSample, Sample, SampleFormat, Stream, StreamConfig, I24};
use openhoshimi_core::{InputSource, IoError, IqSample, IqSource};

const DEFAULT_BUFFER_SAMPLES: usize = 48_000;

/// How `build_stream` should pull samples out of an interleaved frame of
/// per-channel data.
///
/// `Mono` keeps the first channel of each frame and discards the rest,
/// matching the behaviour expected by `InputSource` consumers.  `IqStereo`
/// requires the device to be configured with at least two channels and
/// pairs channel 0 (I) with channel 1 (Q), pushing the pair into the
/// shared buffer as two consecutive f32 values that the IQ source then
/// recombines into [`IqSample`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SamplingMode {
    Mono,
    IqStereo,
}

/// Description of a host-enumerated input device, suitable for populating
/// a UI dropdown.
#[derive(Debug, Clone)]
pub struct SoundcardDeviceInfo {
    /// Human-readable device name as reported by the audio host.
    pub name: String,
    /// Whether the audio host marks this device as the default input.
    pub is_default: bool,
}

/// Enumerate input devices on the platform's default audio host.
///
/// The returned list is best-effort: devices that error out during
/// enumeration are skipped rather than failing the whole call, so the
/// UI can still pick a working device when one specific entry is in a
/// bad state. The default input device (if any) is marked first.
#[must_use]
pub fn enumerate_input_devices() -> Vec<SoundcardDeviceInfo> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let mut devices: Vec<SoundcardDeviceInfo> = Vec::new();
    let Ok(iter) = host.input_devices() else {
        return Vec::new();
    };
    for device in iter {
        let Ok(name) = device.name() else {
            continue;
        };
        let is_default = default_name.as_deref() == Some(name.as_str());
        devices.push(SoundcardDeviceInfo { name, is_default });
    }
    devices.sort_by(|a, b| match (a.is_default, b.is_default) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.cmp(&b.name),
    });
    devices
}

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

    /// Open the input device whose host-reported name matches `name`.
    ///
    /// The match is exact and case-sensitive: the caller is expected to
    /// use a name returned by [`enumerate_input_devices`]. The device's
    /// own default input configuration is used.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Format`] if no input device with that name is
    /// found on the default audio host, the device has no supported
    /// input configuration, or the input stream cannot be built.
    pub fn open_by_name(name: &str) -> Result<Self, IoError> {
        let host = cpal::default_host();
        let iter = host
            .input_devices()
            .map_err(|err| IoError::Format(format!("input device enumeration failed: {err}")))?;
        let device = iter
            .filter_map(|device| match device.name() {
                Ok(device_name) if device_name == name => Some(device),
                _ => None,
            })
            .next()
            .ok_or_else(|| IoError::Format(format!("no input device named {name:?}")))?;
        let supported = device.default_input_config().map_err(map_cpal_error)?;
        let sample_rate = supported.sample_rate().0;
        let sample_format = supported.sample_format();
        let channels = usize::from(supported.channels());
        Self::open_device(
            device,
            supported.into(),
            sample_format,
            channels,
            name.to_string(),
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
            SamplingMode::Mono,
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

/// Live soundcard IQ input source backed by `cpal`.
///
/// The device must expose at least two input channels.  Channel 0 is
/// taken as the I component and channel 1 as the Q component, matching
/// the convention used by SDR software (SDR#, GQRX, CubicSDR, …) when
/// it routes baseband IQ through a virtual stereo cable into the
/// system's input.  Extra channels, if any, are ignored.
pub struct SoundcardIqSource {
    state: Arc<(Mutex<SoundcardState>, Condvar)>,
    _stream: Stream,
    sample_rate: u32,
    description: String,
    closed: bool,
}

impl SoundcardIqSource {
    /// Open the host's default input device as a stereo IQ source.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Format`] if there is no default input device,
    /// the device's default input config has fewer than two channels,
    /// or the input stream cannot be built.
    pub fn open_default() -> Result<Self, IoError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| IoError::Format("no default input device available".to_string()))?;
        let supported = device.default_input_config().map_err(map_cpal_error)?;
        let name = device.name().map_err(map_cpal_error)?;
        Self::open_supported(device, supported, name)
    }

    /// Open the input device whose host-reported name matches `name` as
    /// a stereo IQ source.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Format`] if no matching device exists, the
    /// device has fewer than two channels, or the stream cannot be
    /// built.
    pub fn open_by_name(name: &str) -> Result<Self, IoError> {
        let host = cpal::default_host();
        let iter = host
            .input_devices()
            .map_err(|err| IoError::Format(format!("input device enumeration failed: {err}")))?;
        let device = iter
            .filter_map(|device| match device.name() {
                Ok(device_name) if device_name == name => Some(device),
                _ => None,
            })
            .next()
            .ok_or_else(|| IoError::Format(format!("no input device named {name:?}")))?;
        let supported = device.default_input_config().map_err(map_cpal_error)?;
        Self::open_supported(device, supported, name.to_string())
    }

    fn open_supported(
        device: cpal::Device,
        supported: cpal::SupportedStreamConfig,
        name: String,
    ) -> Result<Self, IoError> {
        let channels = usize::from(supported.channels());
        if channels < 2 {
            return Err(IoError::Format(format!(
                "soundcard IQ requires at least 2 channels, device {name:?} reports {channels}"
            )));
        }
        let sample_rate = supported.sample_rate().0;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();
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
            SamplingMode::IqStereo,
            Arc::clone(&state),
        )?;
        stream.play().map_err(map_cpal_error)?;
        Ok(Self {
            state,
            _stream: stream,
            sample_rate,
            description: format!("soundcard IQ {name} ({sample_rate} Hz, stereo)"),
            closed: false,
        })
    }
}

impl IqSource for SoundcardIqSource {
    fn read_samples(&mut self, buf: &mut [IqSample]) -> Result<usize, IoError> {
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

        // Wait until at least one full IQ pair is available.
        while guard.buffer.len() < 2 && !guard.closed && guard.error.is_none() {
            guard = cvar
                .wait(guard)
                .map_err(|_| IoError::Format("soundcard wait failed".to_string()))?;
        }

        if let Some(err) = guard.error.take() {
            self.closed = true;
            return Err(err);
        }
        if guard.buffer.len() < 2 && guard.closed {
            self.closed = true;
            return Err(IoError::EndOfStream);
        }

        let mut written = 0usize;
        while written < buf.len() && guard.buffer.len() >= 2 {
            // pop_front is guaranteed to return Some here.
            let i = guard.buffer.pop_front().unwrap_or(0.0);
            let q = guard.buffer.pop_front().unwrap_or(0.0);
            buf[written] = IqSample { i, q };
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
    mode: SamplingMode,
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
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::I16 => {
                    if let Some(samples) = data.as_slice::<i16>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::I24 => {
                    if let Some(samples) = data.as_slice::<I24>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::I32 => {
                    if let Some(samples) = data.as_slice::<i32>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::I64 => {
                    if let Some(samples) = data.as_slice::<i64>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::U8 => {
                    if let Some(samples) = data.as_slice::<u8>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::U16 => {
                    if let Some(samples) = data.as_slice::<u16>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::U32 => {
                    if let Some(samples) = data.as_slice::<u32>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::U64 => {
                    if let Some(samples) = data.as_slice::<u64>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::F32 => {
                    if let Some(samples) = data.as_slice::<f32>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
                    }
                }
                SampleFormat::F64 => {
                    if let Some(samples) = data.as_slice::<f64>() {
                        push_samples(samples, channels, mode, Arc::clone(&data_state));
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

fn push_samples<T>(
    data: &[T],
    channels: usize,
    mode: SamplingMode,
    state: Arc<(Mutex<SoundcardState>, Condvar)>,
) where
    T: Copy,
    f32: FromSample<T>,
{
    let (lock, cvar) = &*state;
    let Ok(mut guard) = lock.lock() else {
        return;
    };
    let channels = channels.max(1);
    match mode {
        SamplingMode::Mono => {
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
        }
        SamplingMode::IqStereo => {
            // IqStereo expects channels >= 2; the caller validates that.
            // Use channel 0 as I and channel 1 as Q, ignore extras.
            if channels < 2 {
                return;
            }
            for frame in data.chunks_exact(channels) {
                let i = f32::from_sample(frame[0]);
                let q = f32::from_sample(frame[1]);
                // Drop the oldest IQ pair (two f32s) when full so we
                // never split a pair across a wraparound.
                while guard.buffer.len() + 2 > DEFAULT_BUFFER_SAMPLES {
                    guard.buffer.pop_front();
                }
                guard.buffer.push_back(i);
                guard.buffer.push_back(q);
            }
        }
    }
    cvar.notify_all();
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
        push_samples(
            &[0.25f32, -0.25, 0.5, -0.5],
            1,
            SamplingMode::Mono,
            Arc::clone(&state),
        );

        let (lock, _) = &*state;
        let guard = lock.lock().unwrap_or_else(|_| panic!("lock"));
        assert_eq!(guard.buffer.len(), 4);
        assert_eq!(guard.buffer[0], 0.25);
        assert_eq!(guard.buffer[1], -0.25);
        assert_eq!(guard.buffer[2], 0.5);
        assert_eq!(guard.buffer[3], -0.5);
    }
}
