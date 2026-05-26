//! Minimal radix-2 Cooley–Tukey FFT used by frequency estimators and
//! diagnostic tools.
//!
//! The implementation is iterative, allocation-free for the transform
//! itself, and works on power-of-two buffers of [`Complex`] samples.

use std::ops::{Add, Mul, Sub};

/// Complex number used by the FFT.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    /// Real component.
    pub re: f32,
    /// Imaginary component.
    pub im: f32,
}

impl Complex {
    /// Create a complex number from its real and imaginary components.
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Squared magnitude.
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, other: Complex) -> Complex {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

impl Sub for Complex {
    type Output = Complex;
    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

/// Compute the in-place radix-2 FFT of a power-of-two buffer.
///
/// # Panics
///
/// Panics if `buf.len()` is not a power of two.
pub fn fft_in_place(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two");
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = -std::f32::consts::TAU / len as f32;
        let wlen = Complex::new(theta.cos(), theta.sin());
        let mut i = 0usize;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[i + k];
                let t = buf[i + k + half] * w;
                buf[i + k] = u + t;
                buf[i + k + half] = u - t;
                w = w * wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Map FFT bin index to a signed frequency for a sampled signal.
///
/// Bins above the Nyquist index alias to negative frequencies.
pub fn bin_frequency_hz(bin: usize, fft_size: usize, sample_rate: u32) -> f32 {
    let n = fft_size as i32;
    let signed = if (bin as i32) < n / 2 {
        bin as i32
    } else {
        bin as i32 - n
    };
    signed as f32 * sample_rate as f32 / n as f32
}

/// Generate a Hann window of length `n`.
pub fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|k| {
            let theta = std::f32::consts::TAU * k as f32 / (n - 1) as f32;
            0.5 - 0.5 * theta.cos()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_resolves_complex_tone() {
        let n = 1024usize;
        let mut buf: Vec<Complex> = (0..n)
            .map(|k| {
                let theta = std::f32::consts::TAU * 100.0 * k as f32 / n as f32;
                Complex::new(theta.cos(), theta.sin())
            })
            .collect();
        fft_in_place(&mut buf);
        let powers: Vec<f32> = buf.iter().map(|c| c.norm_sqr()).collect();
        let peak_index = powers
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(peak_index, 100);
    }

    #[test]
    fn bin_frequency_handles_aliasing() {
        let sr = 1024;
        assert!((bin_frequency_hz(0, 1024, sr) - 0.0).abs() < 1e-6);
        assert!((bin_frequency_hz(1, 1024, sr) - 1.0).abs() < 1e-6);
        assert!((bin_frequency_hz(1023, 1024, sr) + 1.0).abs() < 1e-6);
    }
}
