use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::Bound;
use std::f64::consts::PI;

// ======================
// Periodogram / spectral density (FFT-based)
// ======================

/// FFT-based periodogram (O(n log n) instead of O(n²))
///
/// Uses rustfft for massive performance gains on large arrays
/// CORRECTED: Matches scipy.signal.periodogram(scaling='spectrum', detrend=False, window='boxcar')
fn periodogram_fft(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // For small n, use DFT (FFT overhead not worth it)
    if n < 64 {
        return periodogram_dft(x);
    }

    use rustfft::{FftPlanner, num_complex::Complex};

    // IMPORTANT: No demeaning (matches detrend=False)
    // Prepare FFT input
    let mut buffer: Vec<Complex<f64>> = x.iter()
        .map(|&v| Complex::new(v, 0.0))  // NO mean subtraction
        .collect();
    
    // Perform FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);
    
    // Extract power spectrum (only positive frequencies)
    let k_max = n / 2;
    let n_f = n as f64;
    
    // SciPy 'spectrum' scaling: |X|² / n²
    let denom = n_f * n_f;
    
    let freqs: Vec<f64> = (0..=k_max).map(|k| k as f64 / n_f).collect();
    
    // Compute power with one-sided doubling
    let mut power: Vec<f64> = buffer[..=k_max].iter()
        .map(|c| c.norm_sqr() / denom)
        .collect();
    
    // One-sided correction for real signals:
    // Double all interior bins (skip DC and Nyquist)
    for k in 1..k_max {
        power[k] *= 2.0;
    }
    // Only double Nyquist if n is odd (no Nyquist bin for odd n)
    if n % 2 == 1 && k_max > 0 {
        power[k_max] *= 2.0;
    }
    
    (freqs, power)
}

/// DFT fallback for small arrays (n < 64)
/// CORRECTED: Matches scipy.signal.periodogram(scaling='spectrum', detrend=False, window='boxcar')
#[inline]
fn periodogram_dft(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let n_f = n as f64;
    let k_max = n / 2;

    let mut freqs = Vec::with_capacity(k_max + 1);
    let mut power = Vec::with_capacity(k_max + 1);

    for k in 0..=k_max {
        let freq = (k as f64) / n_f;
        let mut re = 0.0;
        let mut im = 0.0;

        // IMPORTANT: No demeaning (matches detrend=False)
        for t in 0..n {
            let angle = -2.0 * PI * (k as f64) * (t as f64) / n_f;
            let val = x[t];  // NO mean subtraction
            re += val * angle.cos();
            im += val * angle.sin();
        }

        // SciPy 'spectrum' scaling: |X|² / n²
        let mut p = (re * re + im * im) / (n_f * n_f);
        
        // One-sided correction for real signals
        if k != 0 && !(n % 2 == 0 && k == k_max) {
            p *= 2.0;
        }

        freqs.push(freq);
        power.push(p);
    }

    (freqs, power)
}

/// Raw periodogram (automatically chooses FFT or DFT)
///
/// Python signature:
///     periodogram(x) -> (freqs, power)
#[pyfunction]
pub fn periodogram<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let x = x.as_slice()?;
    let (freqs, power) = periodogram_fft(x);
    let f_arr = PyArray1::from_vec_bound(py, freqs);
    let p_arr = PyArray1::from_vec_bound(py, power);
    Ok((f_arr, p_arr))
}

// ======================
// WELCH'S METHOD (FFT-based)
// ======================

/// Welch's method for power spectral density estimation (FFT-based)
///
/// Python signature:
///     welch_psd(x, nperseg=256, noverlap=None) -> (freqs, psd)
#[pyfunction(signature = (x, nperseg=256, noverlap=None))]
pub fn welch_psd<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nperseg: usize,
    noverlap: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let x_slice = x.as_slice()?;
    let n = x_slice.len();
    
    if n < nperseg {
        // Call periodogram_fft directly instead of periodogram
        let (freqs, power) = periodogram_fft(x_slice);
        return Ok((
            PyArray1::from_vec_bound(py, freqs),
            PyArray1::from_vec_bound(py, power),
        ));
    }
    
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let step = nperseg.saturating_sub(noverlap);
    
    if step == 0 {
        let (freqs, power) = periodogram_fft(x_slice);
        return Ok((
            PyArray1::from_vec_bound(py, freqs),
            PyArray1::from_vec_bound(py, power),
        ));
    }
    
    let n_segments = (n - noverlap) / step;
    if n_segments == 0 {
        let (freqs, power) = periodogram_fft(x_slice);
        return Ok((
            PyArray1::from_vec_bound(py, freqs),
            PyArray1::from_vec_bound(py, power),
        ));
    }
    
    use rustfft::{FftPlanner, num_complex::Complex};
    
    // Setup FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nperseg);
    
    // Precompute Hann window
    let nperseg_f = nperseg as f64;
    let window: Vec<f64> = (0..nperseg)
        .map(|t| 0.5 * (1.0 - (2.0 * PI * t as f64 / nperseg_f).cos()))
        .collect();
    
    // Window power normalization: sum(w^2)
    let win_sumsq: f64 = window.iter().map(|w| w * w).sum::<f64>();

    // Accumulate PSDs
    let k_max = nperseg / 2;
    let mut psd_sum = vec![0.0; k_max + 1];
    let freqs: Vec<f64> = (0..=k_max).map(|k| k as f64 / nperseg_f).collect();

    let mut count = 0;
    for i in 0..n_segments {
        let start = i * step;
        let end = start + nperseg;
        if end > n {
            break;
        }

        let segment = &x_slice[start..end];

        // Demean and apply window
        let mean = segment.iter().sum::<f64>() / nperseg_f;
        let mut buffer: Vec<Complex<f64>> = segment.iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new((s - mean) * w, 0.0))
            .collect();

        // FFT
        fft.process(&mut buffer);

        // Accumulate power with density scaling: |X_k|^2 / (fs * sum(w^2)),
        // fs = 1. Matches scipy.signal.welch(scaling='density').
        let scale = 1.0 / win_sumsq;
        for k in 0..=k_max {
            psd_sum[k] += buffer[k].norm_sqr() * scale;
        }

        count += 1;
    }

    // Average PSDs
    let count_f = count as f64;
    for p in &mut psd_sum {
        *p /= count_f;
    }

    // One-sided spectrum: double every interior bin. DC is never doubled; the
    // last bin is the Nyquist bin only when nperseg is even, in which case it
    // is not doubled either (scipy semantics).
    for k in 1..=k_max {
        if k == k_max && nperseg % 2 == 0 {
            continue;
        }
        psd_sum[k] *= 2.0;
    }

    Ok((
        PyArray1::from_vec_bound(py, freqs),
        PyArray1::from_vec_bound(py, psd_sum),
    ))
}

// ======================
// SPECTRAL HELPER FUNCTIONS
// ======================

/// Cumulative periodogram (FFT-based)
///
/// Python signature:
///     cumulative_periodogram(x) -> (freqs, cumulative_power)
#[pyfunction]
pub fn cumulative_periodogram<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let x = x.as_slice()?;
    let (freqs, power) = periodogram_fft(x);
    
    // Compute cumulative sum
    let mut cumulative = Vec::with_capacity(power.len());
    let mut sum = 0.0;
    for &p in &power {
        sum += p;
        cumulative.push(sum);
    }
    
    // Normalize by total power
    if sum > 0.0 {
        for c in &mut cumulative {
            *c /= sum;
        }
    }
    
    Ok((
        PyArray1::from_vec_bound(py, freqs),
        PyArray1::from_vec_bound(py, cumulative),
    ))
}

/// Dominant frequency detection (FFT-based)
///
/// Python signature:
///     dominant_frequency(x) -> float
#[pyfunction]
pub fn dominant_frequency(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x = x.as_slice()?;
    let (freqs, power) = periodogram_fft(x);
    
    if freqs.len() <= 1 {
        return Ok(f64::NAN);
    }
    
    // Skip DC component
    let max_idx = power[1..].iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i + 1)
        .unwrap_or(1);
    
    Ok(freqs[max_idx])
}

/// Spectral entropy (FFT-based)
///
/// Python signature:
///     spectral_entropy(x) -> float
#[pyfunction]
pub fn spectral_entropy(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x = x.as_slice()?;
    let (_, power) = periodogram_fft(x);
    
    if power.is_empty() {
        return Ok(f64::NAN);
    }
    
    let total_power: f64 = power.iter().sum();
    
    if total_power <= 0.0 {
        return Ok(f64::NAN);
    }
    
    let mut entropy = 0.0;
    for &p in &power {
        if p > 0.0 {
            let prob = p / total_power;
            entropy -= prob * prob.ln();
        }
    }
    
    Ok(entropy)
}

// ======================
// ADDITIONAL SPECTRAL FUNCTIONS
// ======================

/// Bartlett's method (non-overlapping segments, similar to Welch but simpler)
///
/// Python signature:
///     bartlett_psd(x, nperseg=256) -> (freqs, psd)
#[pyfunction(signature = (x, nperseg=256))]
pub fn bartlett_psd<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    nperseg: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    // Bartlett = Welch with 0 overlap
    welch_psd(py, x, nperseg, Some(0))
}

/// Peak finding in spectrum (returns top N frequencies)
///
/// Python signature:
///     spectral_peaks(x, n_peaks=5, min_height=0.0) -> (freqs, powers)
#[pyfunction(signature = (x, n_peaks=5, min_height=0.0))]
pub fn spectral_peaks<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    n_peaks: usize,
    min_height: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let x = x.as_slice()?;
    let (freqs, power) = periodogram_fft(x);
    
    if freqs.len() <= 1 {
        return Ok((
            PyArray1::from_vec_bound(py, vec![]),
            PyArray1::from_vec_bound(py, vec![]),
        ));
    }
    
    // Find local maxima (skip DC component)
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    
    for i in 2..(power.len() - 1) {
        if power[i] > power[i-1] && power[i] > power[i+1] && power[i] >= min_height {
            peaks.push((i, power[i]));
        }
    }
    
    // Sort by power (descending)
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Take top N
    let n_take = n_peaks.min(peaks.len());
    let peak_freqs: Vec<f64> = peaks[..n_take].iter().map(|(i, _)| freqs[*i]).collect();
    let peak_powers: Vec<f64> = peaks[..n_take].iter().map(|(_, p)| *p).collect();
    
    Ok((
        PyArray1::from_vec_bound(py, peak_freqs),
        PyArray1::from_vec_bound(py, peak_powers),
    ))
}

/// Spectral flatness (measure of white-noise-ness)
///
/// Returns ratio of geometric mean to arithmetic mean of power spectrum
/// Value near 1 = white noise, value near 0 = tonal/periodic
///
/// Python signature:
///     spectral_flatness(x) -> float
#[pyfunction]
pub fn spectral_flatness(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x = x.as_slice()?;
    let (_, power) = periodogram_fft(x);
    
    if power.is_empty() {
        return Ok(f64::NAN);
    }
    
    // Filter out zeros and very small values
    let valid_power: Vec<f64> = power.iter()
        .filter(|&&p| p > 1e-10)
        .copied()
        .collect();
    
    if valid_power.is_empty() {
        return Ok(f64::NAN);
    }
    
    let n = valid_power.len() as f64;
    
    // Geometric mean (use log to avoid overflow)
    let log_geo_mean = valid_power.iter().map(|p| p.ln()).sum::<f64>() / n;
    let geo_mean = log_geo_mean.exp();
    
    // Arithmetic mean
    let arith_mean = valid_power.iter().sum::<f64>() / n;
    
    if arith_mean <= 0.0 {
        return Ok(f64::NAN);
    }
    
    Ok(geo_mean / arith_mean)
}

/// Band power (integrate power in frequency band)
///
/// Python signature:
///     band_power(x, freq_low=0.0, freq_high=0.5) -> float
#[pyfunction(signature = (x, freq_low=0.0, freq_high=0.5))]
pub fn band_power(
    x: PyReadonlyArray1<f64>,
    freq_low: f64,
    freq_high: f64,
) -> PyResult<f64> {
    let x = x.as_slice()?;
    let (freqs, power) = periodogram_fft(x);
    
    if freqs.is_empty() {
        return Ok(f64::NAN);
    }
    
    // Find indices in frequency band
    let mut total_power = 0.0;
    for (i, &freq) in freqs.iter().enumerate() {
        if freq >= freq_low && freq <= freq_high {
            total_power += power[i];
        }
    }
    
    Ok(total_power)
}

/// Spectral centroid (center of mass of spectrum)
///
/// Python signature:
///     spectral_centroid(x) -> float
#[pyfunction]
pub fn spectral_centroid(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x = x.as_slice()?;
    let (freqs, power) = periodogram_fft(x);
    
    if freqs.is_empty() {
        return Ok(f64::NAN);
    }
    
    let total_power: f64 = power.iter().sum();
    
    if total_power <= 0.0 {
        return Ok(f64::NAN);
    }
    
    let weighted_sum: f64 = freqs.iter()
        .zip(power.iter())
        .map(|(f, p)| f * p)
        .sum();
    
    Ok(weighted_sum / total_power)
}

/// Spectral rolloff (frequency below which X% of power is contained)
///
/// Python signature:
///     spectral_rolloff(x, percentile=0.85) -> float
#[pyfunction(signature = (x, percentile=0.85))]
pub fn spectral_rolloff(
    x: PyReadonlyArray1<f64>,
    percentile: f64,
) -> PyResult<f64> {
    let x = x.as_slice()?;
    let (freqs, power) = periodogram_fft(x);
    
    if freqs.is_empty() {
        return Ok(f64::NAN);
    }
    
    let total_power: f64 = power.iter().sum();
    
    if total_power <= 0.0 {
        return Ok(f64::NAN);
    }
    
    let threshold = total_power * percentile;
    let mut cumsum = 0.0;
    
    for (i, &p) in power.iter().enumerate() {
        cumsum += p;
        if cumsum >= threshold {
            return Ok(freqs[i]);
        }
    }
    
    Ok(freqs[freqs.len() - 1])
}
